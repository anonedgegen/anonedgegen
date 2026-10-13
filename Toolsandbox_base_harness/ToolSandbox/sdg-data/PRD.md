# Product Requirements Document: Personal Device Assistant Agent

## 1. Overview

The Personal Device Assistant Agent is a conversational AI agent that helps users manage their mobile device's core functions over multi-turn chat. It can **manage contacts**, **send and search messages**, **set and modify reminders**, **control device settings** (cellular, wifi, location service, battery mode), **perform time/unit calculations**, and **search for real-world information** (weather, stocks, locations, holidays, currency conversion) — all subject to device state constraints.

The agent interacts with users via natural language, queries and modifies the device's local databases and services through a defined set of tools, and must ask for clarification when a user request is ambiguous rather than assuming values.

---

## 2. Agent Behavior

### 2.1 System Prompt

The agent receives the following core instructions:

```
Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.
```

### 2.2 Core Behavioral Rules

- The agent must **not assume values** for tool arguments. If information is missing or ambiguous, it must ask the user for clarification.
- Before any write operation (adding, modifying, removing contacts/reminders, sending messages, toggling settings), the agent must **confirm with the user** before proceeding.
- The agent must **not fabricate information** — it should only use data retrieved from its tools.
- The agent must **handle errors gracefully** — when a tool call raises an exception (e.g., `ConnectionError`, `PermissionError`, `ValueError`), the agent should interpret the error, potentially take corrective action (e.g., enable a required service), and retry or inform the user.
- The agent must **recognize when it cannot complete a task** given the available tools and information, and inform the user accordingly rather than hallucinating results.
- The agent makes **one tool call at a time** and cannot send a message and make a tool call simultaneously.

---

## 3. Device State and Service Dependencies

The device has four service toggles that govern which tools are available. The agent must understand and navigate these dependencies.

### 3.1 Service Hierarchy

```
Low Battery Mode
  ├── Cellular Service
  ├── WiFi
  └── Location Service
```

**Low battery mode** is the master switch:
- Turning **on** low battery mode automatically **disables** cellular, wifi, and location service.
- While low battery mode is **on**, cellular, wifi, and location service **cannot be re-enabled** (raises `PermissionError`).
- To restore any dependent service, the user must first allow the agent to **turn off low battery mode**.

### 3.2 Service-to-Tool Dependencies

| Service Required | Tools That Depend On It |
|-----------------|------------------------|
| **Cellular** | `send_message_with_phone_number` |
| **WiFi** | `search_holiday`, `search_lat_lon`, `search_location_around_lat_lon`, `search_weather_around_lat_lon`, `search_stock`, `convert_currency` |
| **Location Service** | `get_current_location`, and `search_location_around_lat_lon` / `search_weather_around_lat_lon` when latitude/longitude are not explicitly provided |

### 3.3 Error Behavior

- Calling a tool without its required service raises `ConnectionError` (cellular, wifi) or `PermissionError` (location service).
- Attempting to toggle a setting that is already in the desired state raises `ValueError`.
- Attempting to enable a service while low battery mode is on raises `PermissionError`.

### 3.4 Default Device State

| Setting | Default |
|---------|---------|
| Cellular | On |
| WiFi | On |
| Location Service | On |
| Low Battery Mode | Off |
| Current Location | 37.334606°N, 122.009102°W |

---

## 4. Domain Rules

### 4.1 Contact Management

**Contacts** have:
- `person_id` (UUID, system-assigned)
- `name`, `phone_number`, `relationship`, `is_self` flag

**Rules:**
1. There can be **at most one** contact with `is_self=True` (the device owner). Attempting to add a second self entry raises `DuplicateError`.
2. Contacts with identical information **are allowed** — each gets a unique `person_id`.
3. **Modifying** a contact requires at least one changed field; providing no updates raises `ValueError`.
4. **Removing** a contact requires a valid `person_id`; an invalid ID raises `NoDataError`.
5. **Searching** contacts uses:
   - **Exact matching** for `person_id`, `phone_number`, `is_self`
   - **Fuzzy string matching** for `name` (default threshold) and `relationship` (threshold=90)
6. Phone numbers are validated using international phone number parsing. Invalid formats raise `NumberParseException`.

### 4.2 Messaging

**Messages** have:
- `message_id` (UUID), `sender_person_id`, `sender_phone_number`, `recipient_person_id`, `recipient_phone_number`, `content`, `creation_timestamp`

**Rules:**
1. Sending a message **requires cellular service** to be enabled.
2. The sender is automatically resolved from the **self contact entry** — exactly one self entry must exist.
3. The recipient does not need to be in the contact database.
4. **Searching** messages uses:
   - **Exact matching** for `message_id`, `sender_person_id`, `sender_phone_number`, `recipient_person_id`, `recipient_phone_number`
   - **Fuzzy string matching** for `content` (threshold=50)
   - **Range matching** for `creation_timestamp` with lowerbound/upperbound

### 4.3 Reminders

**Reminders** have:
- `reminder_id` (UUID), `content`, `creation_timestamp`, `reminder_timestamp`, `latitude`, `longitude`

**Rules:**
1. `reminder_timestamp` specifies when the user should be reminded, as a POSIX timestamp.
2. `latitude` and `longitude` are optional location fields for location-based reminders.
3. **Modifying** a reminder requires at least one changed field. The `creation_timestamp` is automatically updated on modification.
4. **Searching** reminders uses:
   - **Exact matching** for `reminder_id`, `latitude`, `longitude`
   - **Fuzzy string matching** for `content` (threshold=50)
   - **Range matching** for `creation_timestamp` and `reminder_timestamp` with lowerbound/upperbound

### 4.4 Timestamp and Unit Handling

1. All timestamps are **POSIX timestamps** (seconds since epoch).
2. Timestamps are validated to be between **Jan 1, 1980** and **Jan 1, 2050** (in seconds). Values outside this range (e.g., millisecond-precision timestamps) raise `ValueError`.
3. The agent has tools to convert between POSIX timestamps and human-readable date/time components, shift timestamps, compute differences, and convert units.
4. **Holiday search** returns POSIX timestamps for US holidays only.
5. **Unit conversion** accepts common English unit names (e.g., "celsius", "fahrenheit", "miles", "kilometers").
6. **Currency conversion** requires **ISO 4217** currency codes (e.g., "USD", "EUR").

### 4.5 Location and Geographic Data

1. Latitude must be in range **[-90.0, 90.0]** degrees.
2. Longitude must be in range **[-180.0, 180.0]** degrees.
3. When `search_location_around_lat_lon` or `search_weather_around_lat_lon` are called without explicit latitude/longitude, they default to the **current device location** — which requires location service to be enabled.
4. If only one of latitude/longitude is provided (but not both), a `ValueError` is raised.
5. Geographic distance between coordinates is computed using geodesic distance (in kilometers).

---

## 5. Tools

### 5.1 Contact Tools

| Tool | Type | Description |
|------|------|-------------|
| `add_contact` | write | Add a new contact with name, phone number, optional relationship and is_self flag |
| `modify_contact` | write | Update one or more fields on an existing contact by person_id |
| `remove_contact` | write | Delete a contact by person_id |
| `search_contacts` | read | Search contacts by any combination of fields (fuzzy name/relationship, exact others) |

### 5.2 Messaging Tools

| Tool | Type | Description |
|------|------|-------------|
| `send_message_with_phone_number` | write | Send a message to a phone number. Requires cellular service. |
| `search_messages` | read | Search messages by sender, recipient, content, or timestamp range |

### 5.3 Reminder Tools

| Tool | Type | Description |
|------|------|-------------|
| `add_reminder` | write | Create a reminder with content, time, and optional location |
| `modify_reminder` | write | Update fields on an existing reminder by reminder_id |
| `search_reminder` | read | Search reminders by content, timestamps, or location |
| `remove_reminder` | write | Delete a reminder by reminder_id |

### 5.4 Settings Tools

| Tool | Type | Description |
|------|------|-------------|
| `set_low_battery_mode_status` | write | Toggle low battery mode on/off. Turning on disables all dependent services. |
| `get_low_battery_mode_status` | read | Check if low battery mode is on |
| `set_cellular_service_status` | write | Toggle cellular on/off. Blocked if low battery mode is on. |
| `get_cellular_service_status` | read | Check if cellular is on |
| `set_wifi_status` | write | Toggle wifi on/off. Blocked if low battery mode is on. |
| `get_wifi_status` | read | Check if wifi is on |
| `set_location_service_status` | write | Toggle location service on/off. Blocked if low battery mode is on. |
| `get_location_service_status` | read | Check if location service is on |
| `get_current_location` | read | Get current latitude/longitude. Requires location service. |

### 5.5 Utility Tools

| Tool | Type | Description |
|------|------|-------------|
| `get_current_timestamp` | generic | Get current POSIX timestamp |
| `timestamp_to_datetime_info` | generic | Convert POSIX timestamp to year/month/day/hour/minute/second/isoweekday |
| `datetime_info_to_timestamp` | generic | Convert date/time components to POSIX timestamp |
| `shift_timestamp` | generic | Shift a POSIX timestamp by weeks/days/hours/minutes/seconds |
| `timestamp_diff` | generic | Compute difference between two timestamps as days and seconds |
| `seconds_to_hours_minutes_seconds` | generic | Convert seconds to hour/minute/second components |
| `unit_conversion` | generic | Convert between units (e.g., celsius to fahrenheit, miles to km) |
| `calculate_lat_lon_distance` | generic | Compute geodesic distance (km) between two coordinate pairs |
| `search_holiday` | read | Search for a US holiday by name and optional year. Requires wifi. |

### 5.6 Web-Backed Search Tools (Require WiFi)

| Tool | Type | Description |
|------|------|-------------|
| `search_lat_lon` | read | Reverse geocode: latitude/longitude → address string |
| `search_location_around_lat_lon` | read | Search for businesses/POIs near a location. Defaults to current location if lat/lon omitted. |
| `search_weather_around_lat_lon` | read | Get weather forecast near a location (today or future days). Defaults to current location if lat/lon omitted. |
| `search_stock` | read | Search stock info by company name, symbol, or exchange |
| `convert_currency` | read | Convert currency amounts using ISO 4217 codes |

### 5.7 Control Tools

| Tool | Type | Description |
|------|------|-------------|
| `done` | control | Agent signals task completion. Returns stop token `###STOP###`. |

---

## 6. Data Model

### 6.1 Contact

| Field | Type | Description |
|-------|------|-------------|
| `person_id` | string | UUID, system-assigned |
| `name` | string | Contact's name |
| `phone_number` | string | International format phone number |
| `relationship` | string (nullable) | Relationship to device owner (e.g., "friend", "boss") |
| `is_self` | boolean | True if this contact is the device owner |

### 6.2 Message

| Field | Type | Description |
|-------|------|-------------|
| `message_id` | string | UUID, system-assigned |
| `sender_person_id` | string | Sender's contact person_id |
| `sender_phone_number` | string | Sender's phone number |
| `recipient_person_id` | string (nullable) | Recipient's person_id (null if not in contacts) |
| `recipient_phone_number` | string | Recipient's phone number |
| `content` | string | Message text |
| `creation_timestamp` | number | POSIX timestamp of message creation |

### 6.3 Reminder

| Field | Type | Description |
|-------|------|-------------|
| `reminder_id` | string | UUID, system-assigned |
| `content` | string | Reminder text |
| `creation_timestamp` | number | POSIX timestamp of when the reminder was created |
| `reminder_timestamp` | number | POSIX timestamp of when the user should be reminded |
| `latitude` | number (nullable) | Optional location latitude |
| `longitude` | number (nullable) | Optional location longitude |

### 6.4 Location

| Field | Type | Description |
|-------|------|-------------|
| `latitude` | number | Latitude in degrees (-90 to 90) |
| `longitude` | number | Longitude in degrees (-180 to 180) |

### 6.5 Default Database Contents

| Entity | Default Count |
|--------|--------------|
| Contacts | 4 (1 self, 3 others) |
| Messages | 5 |
| Reminders | 3 |

---

## 7. Key Design Constraints

1. **State dependencies are not bypassed.** Tools that require a service (cellular, wifi, location) will raise errors if that service is off. The agent must detect these errors and either enable the required service (with user permission) or inform the user.

2. **One action per turn.** The agent cannot send a message and make a tool call in the same turn.

3. **No assumptions.** The agent must not guess values for function parameters. If a user asks to "send a message to my friend" but has multiple friends, the agent must ask which one.

4. **Fuzzy vs. exact matching.** Search tools use different matching strategies per field. The agent should understand that name and content searches are approximate, while ID and phone number searches are exact.

5. **POSIX timestamps only.** All time-related operations use POSIX timestamps in seconds. The agent must use the provided conversion tools (`timestamp_to_datetime_info`, `datetime_info_to_timestamp`, `shift_timestamp`, etc.) to translate between human-readable dates and timestamps — it should not attempt to compute timestamps manually.

6. **Validation is enforced at the tool level.** Phone numbers, timestamps, latitude, longitude, and currency codes are validated on every call. Invalid inputs produce descriptive errors that the agent should relay to the user.

7. **Maximum 30 messages per conversation.** Conversations are capped at 30 total messages before termination.

8. **Write tools require confirmation.** The agent must describe the intended action and receive explicit user confirmation before calling any tool that modifies state.
