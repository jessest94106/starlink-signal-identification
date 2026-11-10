Add precomputed prediction files here as `.json` files. Each file contains one record or a list of records:

```json
{
  "satellite_id": "49745",
  "satellite_name": "EXAMPLE-SATELLITE",
  "timestamps": [1791072000.0, 1791072001.0, 1791072002.0],
  "predicted_doppler_hz": [90000.0, 88000.0, 86000.0],
  "azimuth_deg": [110.0, 111.0, 112.0],
  "elevation_deg": [30.0, 31.0, 32.0]
}
```

Timestamps are **Unix seconds in UTC**, strictly increasing. Arrays must be finite and equal in length. Predictions must use the same frequency reference as the measured offsets: the configured receiver center frequency. If a predictor reports Doppler relative to a carrier, add `carrier_hz - center_frequency_hz` before exporting. RF/IF conversion must be consistent with any LNB LO.

The supplied files do not include the existing orbit predictor or trajectory matcher. The implemented matcher interpolates precomputed frequency trajectories and ranks them by RMSE; the default acceptance tolerance is 1,500 Hz and the score is `exp(-RMSE/tolerance)`. This score is a similarity measure, **not a calibrated probability of identification**. At least 30 observations spanning 0.25 seconds are required; matching attempts occur at most once per track every two seconds.

Prediction files are loaded on processing configuration/reset. Malformed files are skipped with errors exposed in `/api/status`. No satellite predictions or identifications are fabricated by the simulator. Export your existing predictor's results to this schema, then reset processing to load them.
