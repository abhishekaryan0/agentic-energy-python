# Client update — Charts that match the recommendation

## What you asked for

1. If the recommendation says the issue is driven by specific sensors, the graph should show those same sensors.
2. The graph should tell the same story as the written recommendation.
3. Clarify why a recommendation dated 8/21 might appear to have graph data through 8/26.

---

## What we changed

### 1) Graph shows the sensors that drive the recommendation

**Before:** Charts sometimes showed a generic model metric (for example reconstruction error), even when the text talked about discharge pressure, motor temperature, drive frequency, injection rate, or water cut.

**Now:**
- We read the recommendation text / contributing-sensor list.
- We put those parameters on the graph first.
- We also add oil/water production for context when helpful.

**Example**

Recommendation says:  
*"The issue is mainly driven by injection rate and water cut."*

Graph now shows:
- Injection Rate
- Water Cut
- Oil Production (context)
- Water Production (context when relevant)

So an engineer can visually check:

Injection ↑ → Water Cut ↑ → Oil ↓

### 2) Same story in text and chart

| Recommendation mentions | Graph prioritizes |
|---|---|
| Discharge pressure, motor temperature, drive frequency | Those three series + oil/water |
| Injection rate and water cut | Injection rate, water cut + oil/water |
| Pump fillage / SPM / intake pressure | Matching sensors + production context |

We also store `highlighted_parameters` in `chart_data` so the UI knows which series are the main drivers.

### 3) Recommendation date vs graph date (your 8/21 vs 8/26 question)

**Clarification**

- The graph is a **snapshot at recommendation generation time**.
- It is **not** continuously updated after the recommendation is created.
- Chart data is clipped to the recommendation / detection date so it should **not** include later “future” days.

We now include snapshot metadata in chart payload, for example:

```json
"snapshot": {
  "recommendation_created_at": "2026-08-21T...",
  "data_as_of": "2026-08-21",
  "continuously_updated": false,
  "note": "Snapshot when the recommendation was generated; not live-updated afterward."
}
```

**Why 8/21 vs 8/26 may have appeared before**
- Older charts could use a data window ending at “latest available in the file,” which might be newer than the recommendation date if data was reloaded later.
- That is what we fixed: chart end date is now tied to the recommendation/detection date.

---

## Short message you can send the client

We updated recommendation charts so they visually support the written finding.

If a recommendation says the issue is driven by discharge pressure, motor temperature, and drive frequency, the graph now shows those sensors (plus oil/water for context)—not a generic reconstruction-error plot.

The same applies when the text cites injection rate and water cut: those series are what appear on the chart so engineers can verify the relationship.

On dates: the graph is a snapshot from when the recommendation was generated. It is not live-updating after creation, and chart data is capped at the recommendation/detection date so it should not include later days.
