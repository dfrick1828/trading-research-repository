# Northern Colorado Water Right Outlook — Public Beta with CDSS API

This version uses:
- CDSS live active administrative calls when enabled
- Historical daily regime data as the model backbone
- Expert water-right rules
- Public-facing Poudre River landing page

## Optional API key

CDSS supports anonymous use but applies call/row limits. For reliability, set a Streamlit secret:

```toml
CDSS_API_KEY = "your-key-here"
```

The app sends the key using the `ApiKey` request header.

## Deploy

Upload all files to GitHub and deploy on Streamlit Cloud with `app.py`.


## Current call selection

The live current-call display now uses the most recently set active call from the included districts, rather than the most severe active call.


## Same-day call-depth percentile

The landing-page "Today's condition" metric now compares the current call-depth/severity score against historical observations for the same day of year, rather than against all days in the record. If exact day-of-year history is sparse, the app falls back to a +/- 7 day seasonal window.


## Latest patch

- The default scenario/historical date now opens on the current calendar day.
- If the historical snapshot does not contain today's exact date, the app uses the most recent historical row with the same day-of-year as a model template.
- Removed the "Area covered by the Outlook" section from the landing page.


## Current call selection rule

The reported/current call is selected as the most senior active call in WD1 and WD3.
Selection uses lowest CDSS Priority Admin No, with Priority Date as a fallback.


## Current Poudre call selection rule

The public-facing current call is selected as the active Cache la Poudre call:
- WD3 records;
- prefer Water Source / Location Structure / Priority Structure fields containing "POUDRE";
- select the most recently set active Poudre call, using Priority Admin No as a tie-breaker.

This is intended to show the current Poudre call, such as the 11/20/1874 call, rather than the most senior call elsewhere in WD1/WD3.


## Latest dashboard change

Removed the "Administrative stress around selected date" chart and replaced it with a 30-day probability outlook chart showing the likelihood of each water-right regime.


## Latest change

Public-facing historical charts now start at 2025.


## Latest change

Annual regime history chart restored to 2005–present.


## Clean stochastic rebuild

Rebuilt from the stable pre-stochastic app. The stochastic UI is inserted only after `daily`, `annual`, `flow`, `model_df`, `clf`, `le`, and `row` exist.


## Latest change

Removed the public landing page. The app now opens directly to the Water Right Outlook dashboard.


## Latest change

Removed the duplicate/second header from the dashboard build.


## Latest change

Removed the user option to disable live CDSS data. The dashboard now always attempts to use live CDSS active calls. The historical data remains the model backbone and emergency fallback if CDSS is unavailable.


## Latest change

- Removed pill/bubble UI chips above dashboard data.
- Active/current call priority date is formatted as MM/DD/YYYY.


## Latest change

Removed/suppressed remaining long horizontal divider/bubble elements above the dashboard data, including hero-footer, badge, pill, status-pill, and metric-row elements.


## Latest change

Removed the rounded top snapshot cards/bubbles above the dashboard data and replaced them with a flat text-based Current Water Right Snapshot.


## Rebuilt no-bubbles version

Rebuilt the app to remove the decorative rounded horizontal capsule/divider elements from the metric sections. Metrics are now rendered as flat text in columns. Live CDSS remains always on, and call dates remain formatted as MM/DD/YYYY.


## Latest change

- Updated public regime labels:
  - Free River
  - Light Administration
  - Active Administration
  - Restrictive Administration
  - Senior Calls Dominating
- Removed the stochastic hydrology section.
- Rebuilt the Annual Regime History chart as a single uniquely-keyed Plotly chart to avoid StreamlitDuplicateElementId errors.


## Latest change

Simplified the top of the dashboard to three primary public-facing metrics:
1. Current Condition
2. Current Call
3. 30-Day Outlook

Flow, comparable years, and severity score were moved into a secondary Supporting Hydrology and Context section.
