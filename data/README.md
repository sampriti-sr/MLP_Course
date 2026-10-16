# Course datasets: provenance & audit

Maintainer-facing record of every dataset the course uses: where it comes from, its licence, how big it is, and what is *deliberately messy* about it. Students should read the friendlier page `docs/getting-started/datasets.md` instead.

**No raw data is committed to this repository.** Files are downloaded on demand by `mlp_data` (see `mlp_data/`) from the original publishers, cached locally, and checked against the SHA-256 values below.

- **Audit date:** 2026-10-15
- **Method:** each URL was requested; licence text was read from the publisher's own page; each file was downloaded and loaded with pandas 3.0 / pyarrow to confirm schema and data-quality claims.
- **Licence summary is not legal advice.** Re-check before redistributing anything.

## Summary

| Key | Dataset | Track | Download size | Rows × cols | Licence | Status |
|---|---|---|---|---|---|---|
| `retail` | UCI Online Retail II | Lifecycle, segmentation, forecasting, recsys | 45.6 MB (xlsx in zip) | 1,067,371 × 8 | CC BY 4.0 | ✅ verified |
| `movielens` | MovieLens `ml-latest-small` | Recommender | 1.0 MB | 100,836 ratings | GroupLens terms (non-commercial; redistribution allowed under same terms) | ✅ verified |
| `taxi` | NYC TLC Yellow Taxi, one file per month | Time series | ~47 MB per month | 3,066,766 × 19 (2023-01) | **No licence stated** by TLC (accuracy disclaimer only) | ⚠️ see note |
| `airbnb` | Inside Airbnb, London `listings.csv.gz`, snapshot 2026-09-17 | Messy tabular regression | 48.3 MB | 91,954 × 90 | CC BY 4.0 | ✅ verified, snapshot rotates |
| `credit` | UCI Default of Credit Card Clients (Taiwan, 2005) | Imbalanced classification, fairness | 5.5 MB (xls in zip) | 30,000 × 25 | CC BY 4.0 | ✅ verified |

## Per-dataset notes

### `retail`: UCI Online Retail II
- **Source:** <https://archive.ics.uci.edu/dataset/502/online+retail+ii> · direct file `https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip`
- **Citation:** Chen, D. (2019). *Online Retail II* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5CG6D>
- **Contents:** two sheets (2009-12 → 2010-11 and 2010-12 → 2011-12) of UK online-retailer invoices; many customers are wholesalers.
- **Verified messiness (this is why we chose it):**
  - 22.8 % of rows have no `Customer ID`.
  - 19,494 cancellation lines (`Invoice` starts with `C`); 22,950 rows with `Quantity <= 0`; 6,207 with `Price <= 0`.
  - 34,335 exact duplicate rows.
  - 5,477 non-product `StockCode` rows (`POST`, `M`, `D`, `ADJUST`, `AMAZONFEE`, …).
  - **Mixed dtypes:** `Invoice`, `StockCode` and `Description` contain both ints and strings, so writing to parquet fails unless columns are cast. The loader casts text columns to pandas `string`.
  - **Orders cancelled almost immediately:** 142 invoices (0.39%), carrying about 2.2% of sales revenue, are fully reversed by a cancellation within a day, including a GBP 77,184 order cancelled sixteen minutes later. They do not change any "bought again" label but they inflate spend, and with it any value-ranked list. `mlp_data.retail.clean_retail` nets each such purchase and its cancellation out (cleaning step 6).
- **Practicalities:** reading the xlsx takes ~35 s, so the loader caches a parquet copy (~7 MB).

### `movielens`: MovieLens small
- **Source:** <https://grouplens.org/datasets/movielens/latest/> · `https://files.grouplens.org/datasets/movielens/ml-latest-small.zip`
- **Terms:** may redistribute (including transformations) under the same conditions; **no commercial or revenue-bearing use** without GroupLens permission; must cite Harper & Konstan (2015), *The MovieLens Datasets: History and Context*, ACM TiiS 5(4).
- **Contents:** `ratings` (userId, movieId, rating, timestamp), `movies`, `links`, `tags`. 610 users; 9,742 movies in the catalogue but only **9,724 have ratings**; **98.3 % sparse**; every user has ≥ 20 ratings (median 70.5).
- **Implication:** the non-commercial clause means this course must stay free/educational; flag if that ever changes. The "small" snapshot is dated 2018-09 and is stable, but `latest-small` is an alias. Pin by checksum.

### `taxi`: NYC TLC Yellow Taxi
- **Source:** <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page> · files at `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet`
- ⚠️ **Licence:** the TLC page states that TLC makes no representations as to accuracy and publishes no licence or reuse terms. The data is also published on NYC Open Data. **Open item:** confirm the Open Data terms before we rely on this for anything other than on-demand download (we never redistribute it, which keeps us safe either way).
- **Verified messiness (2023-01):** 48 trips with pickup outside the month (earliest 2008-12-31); 25,049 negative `fare_amount`; 45,862 zero-distance trips; 71,743 null `passenger_count`. The TLC page warns the parquet schema may be standardised across years, so **pin the months we use** and verify columns.
- **Verified URLs:** 2023-01, 2023-02 and 2024-01 all resolve (HTTP 200). For the time-series track (Phase 4) we also use 2023-02, 2023-03 and 2020-01 to 2020-05 (all HTTP 200; checksums below, recorded on first download, 2026-10-16). `mlp_data.timeseries.taxi_hourly` reads only the pickup-time column and caches hourly counts.
- **Time-series facts (verified):** in 2023-01 to 2023-03 exactly one hourly slot is empty, 02:00 on 2023-03-12 (daylight-saving change; timestamps are naive local time). 2020 months shrink from 6.4 M trips (Jan) to 0.24 M (Apr).

### `airbnb`: Inside Airbnb, London
- **Source:** <https://insideairbnb.com/get-the-data/> · `…/london/2026-09-17/data/listings.csv.gz`
- **Licence:** CC BY 4.0 (stated on the site). Credit "Inside Airbnb".
- ⚠️ **Link rot:** the site keeps only roughly the last year of snapshots, so the pinned URL **will** eventually 404. Policy: pin one snapshot per release; the loader raises a clear error pointing to the data-request page. A mirror (GitHub Release asset, permitted by CC BY with attribution) is the long-term fix; see open items.
- **Verified messiness:** `price` is a *string* (with currency symbol) and **34.9 % null**; `review_scores_rating` 22.5 % null; 90 columns including free text (`description`, `amenities`) and coordinates.
- **Note:** the `visualisations/listings.csv` (summary) variant returned HTTP 403 when requested directly. Use the full `data/listings.csv.gz`.

### `credit`: Default of Credit Card Clients
- **Source:** <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients> · `https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip`
- **Citation:** Yeh, I-C. (2016). UCI Machine Learning Repository. <https://doi.org/10.24432/C55S3H>
- **Contents:** 30,000 credit-card clients in Taiwan (2005); target `default payment next month`; default rate **22.1 %**.
- **Verified messiness:** `EDUCATION` contains undocumented codes 0, 5, 6; `MARRIAGE` contains 0. The header is on the *second* row of the sheet (`header=1`). Contains `SEX` and `AGE`, which makes it suitable for a fairness exercise.
- **Caveat for teaching:** 2005 Taiwan data. Not representative of any current lending population; say so on the page.

## Checksums (SHA-256, as downloaded 2026-10-15)

| File | SHA-256 |
|---|---|
| `online+retail+ii.zip` | `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb` |
| `online_retail_II.xlsx` (inside zip) | `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980` |
| `ml-latest-small.zip` | `696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436` |
| `default+of+credit+card+clients.zip` | `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602` |
| `default of credit card clients.xls` (inside zip) | `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933` |
| `yellow_tripdata_2023-01.parquet` | `32df6f67578fa86c484a6b5ef23a5281992ff085521082340b0f9e5889e9a572` |
| `yellow_tripdata_2023-02.parquet` | `4809e6aaac64f05a62d16a25d55713be1537ad64fc261e895eaf2d2120fe750a` |
| `yellow_tripdata_2023-03.parquet` | `e7d44943111b007bf0e7084863511886e0db29f862f9a96239383a1f86c6c26e` |
| `yellow_tripdata_2020-01.parquet` | `0c32c1d5ef0d37ac3ff1a3f1880f247cc40165edf18626e5a86866296a4a5b93` |
| `yellow_tripdata_2020-02.parquet` | `e57df0e7c410c02d78cd59a31394a118f3d92c50560a2d8fa3a2bf5e17dcffaa` |
| `yellow_tripdata_2020-03.parquet` | `6fa1343946bc1e3702fc351ff9818787cc40e4cf2b095d947ff5e452b5b42f84` |
| `yellow_tripdata_2020-04.parquet` | `3104e1e5548ade630988a4a8b9ec76c51a057b88159f78676c7d344f48491450` |
| `yellow_tripdata_2020-05.parquet` | `f6167e3c8f1039ec4015cb121e73f8e08710ad1681da747f2b5aa504008cf289` |
| `london/2026-09-17/listings.csv.gz` | `8017870734a9bffa1b424937372e5f02f8c4a114e5060534d7423be5fd718b4c` |

## Open items for the maintainers

1. **TLC reuse terms**: confirm via NYC Open Data before we ship any derived/sampled copy.
2. **Mirror** Airbnb (and optionally retail parquet) as a GitHub Release asset to remove link-rot and the 35 s xlsx parse. Needs a decision on repo storage and attribution text.
3. **Additional taxi months**: pick the exact range for the time-series track (Phase 4) and add their checksums.
4. **Credit-default alternative**: if Taiwan-2005 is judged too dated, evaluate a more recent public credit/fraud dataset.
5. **MovieLens** non-commercial clause: acceptable for a free course; revisit if the course is ever monetised.
