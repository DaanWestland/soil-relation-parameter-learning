# Pilot data table (not included)

The pilot used one frozen table, `conus_0_30.parquet`: 28,192 soil profiles of the conterminous USA, one row per
profile, 38 columns (clay, organic carbon, pH, total N, CEC7, CaCO3 equivalent, coordinates, EPSG:5070 block and
region ids, 19 bioclimatic variables, elevation, slope, relief).

SHA-256: `7a890cb6be3d858894c48cd48c9c28c45aedeec42f335a278c9d42a4473ee624`

**Why it is not here.** The table is derived from WoSIS "latest" (ISRIC; every source dataset keeps its own
licence, 27,878 of the 28,192 profiles are from the US National Cooperative Soil Survey compilation) and from
WorldClim 2.1, whose terms allow academic and non-commercial use but not redistribution without permission.
See `NOTICE`.

**Two ways to get a table**

1. *Rebuild it* (anyone): `python -m pedopilot download` and `python -m pedopilot prep`
   (`src/pedopilot/data.py` documents every step). WoSIS "latest" changes over time, so the rebuilt table
   differs somewhat from the pilot's (`check` reports this) and results will be close, not identical.
2. *Ask for the exact table* (researchers): write to the author (see the README). Put the file here as
   `data_snapshot/conus_0_30.parquet`; `python -m pedopilot check` verifies the SHA-256 above.
