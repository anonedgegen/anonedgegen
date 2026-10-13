# Product Requirements Document: Airline Customer Service Agent

## 1. Overview

The Airline Customer Service Agent is a conversational AI agent that assists customers with airline-related requests over multi-turn chat. It can **book flights**, **modify reservations**, **cancel reservations**, and **handle refunds and compensation** — all governed by a strict operational policy.

The agent interacts with customers via natural language, queries and modifies a backend database through a defined set of tools, and must obtain explicit user confirmation before any state-changing operations.

---

## 2. Agent Behavior

### 2.1 System Prompt

The agent receives the following core instructions:

```
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
```

The full airline policy (Section 3) is injected into the agent's system prompt within `<policy>` tags.

### 2.2 Core Behavioral Rules

- The agent must **follow the policy strictly** at all times.
- Before any write operation (booking, modifying, cancelling), the agent must **list the action details and obtain explicit user confirmation** ("yes") to proceed.
- The agent must **not provide information, recommendations, or procedures** not found in the policy or available tools.
- The agent must **deny requests that violate policy**.
- The agent must **transfer to a human agent** if the request cannot be handled within the scope of available actions. To transfer, the agent calls `transfer_to_human_agents` and sends the message: `"YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."`
- The agent makes **one tool call at a time** and cannot send a message and make a tool call simultaneously.

---

## 3. Airline Policy

**Simulated current time:** `2024-05-15 15:00:00 EST`

### 3.1 Domain Basics

**Users** have profiles containing:
- User ID, email, address, date of birth
- Payment methods: **credit card**, **gift card**, **travel certificate**
- Membership level: **regular**, **silver**, **gold**
- Reservation numbers

**Flights** have:
- Flight number, origin, destination, scheduled departure/arrival (local time)
- Per-date status: **available** (can be booked), **delayed** / **on time** (cannot be booked), **flying** (airborne, cannot be booked), **landed** (arrived at destination), **cancelled** (cancelled by airline)
- Per-date seat availability and prices by cabin class

**Cabin classes:** **basic economy**, **economy**, **business**
> Basic economy is its own class, completely distinct from economy.

**Reservations** contain:
- Reservation ID, user ID, trip type (**one way** / **round trip**)
- Flights, passengers, payment methods, created time
- Baggage info, travel insurance status

### 3.2 Booking Rules

1. The agent must first obtain the **user ID** from the customer.
2. Then collect: trip type, origin, destination.
3. **Cabin class** must be the same across all flights in a reservation.
4. **Passengers:**
   - Maximum 5 passengers per reservation
   - Agent collects first name, last name, and date of birth for each
   - All passengers fly the same flights in the same cabin
5. **Payment:**
   - Maximum 1 travel certificate, 1 credit card, and 3 gift cards per reservation
   - Remaining balance on a travel certificate is **not refundable**
   - All payment methods must already be in the user's profile
6. **Checked bag allowance** (per passenger, based on booking user's membership):

   | Membership | Basic Economy | Economy | Business |
   |------------|--------------|---------|----------|
   | Regular | 0 free | 1 free | 2 free |
   | Silver | 1 free | 2 free | 3 free |
   | Gold | 2 free | 3 free | 4 free |

   - Extra bags cost **$50 each**
   - Do not add bags the user does not need
7. **Travel insurance:**
   - $30 per passenger
   - Enables full refund for cancellation due to health or weather reasons
   - Agent should ask if the user wants insurance

### 3.3 Modification Rules

The agent must first obtain the **user ID** and **reservation ID**.

**Change flights:**
- Basic economy flights **cannot be modified**
- Other reservations can be modified without changing origin, destination, or trip type
- Some flight segments can be kept; their prices remain at the original price
- **The API does not enforce these rules** — the agent must verify eligibility before calling the API

**Change cabin:**
- Cannot be changed if any flight in the reservation has already been flown
- All reservations (including basic economy) can change cabin without changing flights
- Cabin must remain the same across all flights in the reservation
- Price difference is charged or refunded accordingly

**Change baggage and insurance:**
- Bags can be **added** but not removed
- Insurance **cannot be added** after initial booking

**Change passengers:**
- Passenger details can be modified, but the **number of passengers cannot change**
- Even a human agent cannot modify the number of passengers

**Payment for modifications:**
- If flights are changed, the user must provide a single gift card or credit card for payment/refund
- The payment method must already be in the user's profile

### 3.4 Cancellation Rules

The agent must first obtain the **user ID**, **reservation ID**, and **reason for cancellation** (change of plan, airline cancelled flight, or other reasons).

**Cannot cancel** if any portion of the flight has already been flown (transfer to human agent needed).

**Can cancel** if any of the following is true:
- The booking was made within the last **24 hours**
- The flight was **cancelled by the airline**
- It is a **business class** flight
- The user has **travel insurance** and the reason is covered by insurance

> **The API does not enforce cancellation rules** — the agent must verify eligibility before calling the API.

**Refund:** Goes to original payment methods within 5–7 business days.

### 3.5 Compensation Rules

- **Do not proactively offer** compensation unless the user explicitly asks.
- Always **confirm the facts** before offering compensation.
- **Eligible customers only:** silver/gold members, passengers with travel insurance, or business class flyers.
- Regular members without insurance flying (basic) economy are **not eligible**.

**Compensation amounts:**

| Situation | Amount |
|-----------|--------|
| Cancelled flight | $100 × number of passengers (as travel certificate) |
| Delayed flight (if user wants to change or cancel) | $50 × number of passengers (as travel certificate) |

- No compensation for any other reason.

---

## 4. Tools

### 4.1 Read Tools (No State Change)

| Tool | Description |
|------|-------------|
| `get_user_details` | Get user profile by user ID (includes payment methods, membership, reservations) |
| `get_reservation_details` | Get full reservation details by reservation ID |
| `list_all_airports` | Returns all 20 supported airports (SFO, JFK, LAX, ORD, DFW, DEN, SEA, ATL, MIA, BOS, PHX, IAH, LAS, MCO, EWR, CLT, MSP, DTW, PHL, LGA) |
| `search_direct_flight` | Search direct flights by origin, destination, and date |
| `search_onestop_flight` | Search one-stop connecting flights by origin, destination, and date |
| `get_flight_status` | Get status of a specific flight on a specific date |

### 4.2 Write Tools (Modify State — Require User Confirmation)

| Tool | Description |
|------|-------------|
| `book_reservation` | Book a new reservation with flights, passengers, and payment. Validates flight availability, seat capacity, and payment balances. Deducts from gift cards and removes used certificates. Max 5 passengers, max 3 reservations per task. |
| `cancel_reservation` | Cancel an entire reservation and reverse payment. **Does not enforce cancellation policy** — agent must verify eligibility first. |
| `update_reservation_flights` | Update flights or cabin on a reservation. Keeps original prices for unchanged segments; charges/refunds price differences. Certificates cannot be used for flight updates. |
| `update_reservation_passengers` | Update passenger information on a reservation. |
| `update_reservation_baggages` | Add baggage to a reservation. Certificates cannot be used for baggage payments. |
| `send_certificate` | Send a travel certificate to a user as compensation. Limited to max 3 per task. |

### 4.3 Utility Tools

| Tool | Description |
|------|-------------|
| `transfer_to_human_agents` | Transfer the customer to a human agent with a summary of the issue |
| `calculate` | Evaluate an arithmetic expression (e.g., for computing price differences) |
| `think` | Append a private thought to the log without changing state or messaging the user |

---

## 5. Data Model

### 5.1 User

| Field | Description |
|-------|-------------|
| `user_id` | Unique identifier (e.g., "mia_li_3668") |
| `name` | Full name |
| `address` | Postal address |
| `email` | Email address |
| `dob` | Date of birth |
| `payment_methods` | List of credit cards, gift cards, and/or travel certificates |
| `saved_passengers` | Pre-saved passenger profiles |
| `membership` | One of: `regular`, `silver`, `gold` |
| `reservations` | List of reservation IDs |

### 5.2 Flight

| Field | Description |
|-------|-------------|
| `flight_number` | Unique identifier (e.g., "HAT001") |
| `origin` | Origin airport code |
| `destination` | Destination airport code |
| `scheduled_departure` | Scheduled departure time (local) |
| `scheduled_arrival` | Scheduled arrival time (local) |
| Per-date records | Status (available/on_time/delayed/flying/landed/cancelled), seat availability by cabin, prices by cabin |

### 5.3 Reservation

| Field | Description |
|-------|-------------|
| `reservation_id` | Unique identifier (e.g., "4WQ150") |
| `user_id` | Owner's user ID |
| `origin` | Trip origin |
| `destination` | Trip destination |
| `flight_type` | `one_way` or `round_trip` |
| `cabin` | `basic_economy`, `economy`, or `business` |
| `flights` | List of flight segments |
| `passengers` | List of passengers (first name, last name, DOB) |
| `payment_history` | Payment methods used |
| `created_at` | Reservation creation timestamp |
| `baggages` | Baggage information |
| `insurance` | Whether travel insurance was purchased |
| `status` | Reservation status |

### 5.4 Payment Methods

| Type | Details |
|------|---------|
| **Credit Card** | Standard credit card on file |
| **Gift Card** | Has a balance; deducted upon use |
| **Travel Certificate** | One-time use; remaining balance is not refundable |

### 5.5 Database Scale

| Entity | Count |
|--------|-------|
| Flights | 300 |
| Users | 500 |
| Reservations | 2,000 |
| Airports | 20 (US domestic) |

---

## 6. Key Design Constraints

1. **Policy is not enforced by the API.** Critical write operations (cancel, modify) do not enforce business rules at the API level. The agent must verify eligibility before calling the tool. This means the agent's policy reasoning is essential — a wrong cancellation call will succeed but produce an incorrect outcome.

2. **One action per turn.** The agent cannot send a message and make a tool call in the same turn. This enforces a clear action-or-communicate pattern.

3. **Explicit confirmation required.** All write operations require the agent to list action details and receive explicit user confirmation before proceeding.

4. **Payment methods must be on file.** The agent cannot accept new payment methods — all must already exist in the user's profile.

5. **20 US airports only.** The system covers domestic US flights across: SFO, JFK, LAX, ORD, DFW, DEN, SEA, ATL, MIA, BOS, PHX, IAH, LAS, MCO, EWR, CLT, MSP, DTW, PHL, LGA.
