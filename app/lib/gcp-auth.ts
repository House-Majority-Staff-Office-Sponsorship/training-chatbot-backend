/**
 * GCP Authentication for non-GCP environments (e.g., Vercel).
 *
 * When running outside GCP (no ADC available), this writes the service account
 * key JSON from the GOOGLE_APPLICATION_CREDENTIALS_JSON env var to a temp file
 * and sets GOOGLE_APPLICATION_CREDENTIALS so the Google client libraries pick it up.
 *
 * Import this module early — before any Google SDK usage.
 */

import { writeFileSync, existsSync } from "fs";

const KEY_PATH = "/tmp/gcp-sa-key.json";

if (
  process.env.GOOGLE_APPLICATION_CREDENTIALS_JSON &&
  !process.env.GOOGLE_APPLICATION_CREDENTIALS
) {
  if (!existsSync(KEY_PATH)) {
    writeFileSync(KEY_PATH, process.env.GOOGLE_APPLICATION_CREDENTIALS_JSON, {
      mode: 0o600,
    });
  }
  process.env.GOOGLE_APPLICATION_CREDENTIALS = KEY_PATH;
}
