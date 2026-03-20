# Integrations, Voice & Image Generation

[Back to Architecture Index](../ARCHITECTURE.md)

---

## Authentication (`auth/auth_provider.py`)

Abstraction layer for identity providers. All routes use the provider-agnostic `verify_token()` and `auth_token_required` decorator.

**Provider selection**: Set `AUTH_PROVIDER` env var:

| Provider | Env Value | Config Required | Notes |
|----------|-----------|-----------------|-------|
| **Firebase** | `firebase` (default) | `FIREBASE_PROJECT_ID` | Lightweight JWT verification (no firebase-admin SDK). |
| **AWS Cognito** | `cognito` | `AWS_COGNITO_USER_POOL_ID`, `AWS_COGNITO_CLIENT_ID`, `AWS_COGNITO_REGION` | Standard OIDC/JWKS verification. |

Both providers verify RS256 JWTs and return normalized claims (`uid`, `email`, `email_verified`). Legacy session token auth (PostgreSQL-based) is always available as a fallback regardless of provider choice.

### Firebase Setup

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Create a project (or use an existing one)
3. Go to **Project Settings** (gear icon) > **General**
4. Copy the **Project ID** (e.g., `my-companion-app`)
5. Go to **Authentication** > **Sign-in method** and enable Email/Password (or any provider you want)
6. Set in `.env`:
   ```
   AUTH_PROVIDER=firebase
   FIREBASE_PROJECT_ID=my-companion-app
   ```
7. In your frontend, use the Firebase JS SDK to get ID tokens and send them as `Authorization: Bearer <token>`

### AWS Cognito Setup

1. Sign in to the [AWS Console](https://console.aws.amazon.com/)
2. Go to **Amazon Cognito** > **User pools** > **Create user pool**
3. Configure sign-in:
   - Select **Email** as a sign-in option
   - Configure password policy and MFA as desired
4. Configure the app client:
   - Go to **App integration** tab > **Create app client**
   - App type: **Public client**
   - App client name: e.g., `companion-web`
   - Authentication flows: select **ALLOW_USER_SRP_AUTH** and **ALLOW_REFRESH_TOKEN_AUTH**
   - Click **Create app client**
5. Copy configuration values:
   - **User pool ID**: Found on the user pool overview page (format: `us-east-1_xxxxxxxxx`)
   - **Client ID**: Found under **App integration** > **App clients** (format: random alphanumeric string)
   - **Region**: The AWS region where you created the pool (e.g., `us-east-1`)
6. Set in `.env`:
   ```
   AUTH_PROVIDER=cognito
   AWS_COGNITO_USER_POOL_ID=us-east-1_xxxxxxxxx
   AWS_COGNITO_CLIENT_ID=xxxxxxxxxxxxxxxxxxxxxxxxxx
   AWS_COGNITO_REGION=us-east-1
   ```
7. In your frontend, use the AWS Amplify SDK or `amazon-cognito-identity-js` to get ID tokens and send them as `Authorization: Bearer <token>`

### Authentication Flow

```
Client gets ID token (Firebase SDK / Cognito SDK / Amplify)
  -> Sends: Authorization: Bearer <id_token>
  -> Backend: auth_provider.verify_token()
     -> Firebase: fetches Google X509 certs, verifies RS256 JWT
     -> Cognito: fetches JWKS from Cognito endpoint, verifies RS256 JWT
  -> Returns normalized claims: {uid, email, email_verified}
  -> Falls back to legacy session token if provider verification fails
```

### Auth Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/verify` | POST | Verify ID token, return user info (provider-agnostic) |
| `/api/auth/user-info` | GET | Get authenticated user info |
| `/api/auth/token-logout` | POST | Server-side logout notification |
| `/api/auth/firebase/verify` | POST | Legacy Firebase-specific verify (delegates to above) |
| `/api/auth/login` | POST | Legacy email/password login |
| `/api/auth/register` | POST | Legacy email/password registration |

---

## Google Service (`integrations/google_service.py`)

OAuth2-authenticated client for Google APIs:

| API | Methods | Scopes |
|-----|---------|--------|
| Gmail | `send_email()`, `search()`, `read()` | `gmail.send`, `gmail.readonly` |
| Calendar | `create_calendar_event()`, read events | `calendar`, `calendar.events` |
| Docs | `create_document()` | `documents` |

Credentials stored at `credentials/google_token.json` with auto-refresh on expiry.

## Calendar Service (`integrations/calendar_service.py`)

Read-side facade for Google Calendars with 5-minute cache. Fetches from all user calendars, merges and sorts. Supports dedicated companion calendar via `companion_calendar_id.txt`.

Key helpers: `time_until_next_event()`, `is_user_busy()`, `get_upcoming_events()`.

## Telegram Bridge (`autonomy/telegram_bridge.py`)

Two-way messaging channel. Uses direct Telegram HTTP API (not python-telegram-bot, to avoid asyncio conflicts with eventlet).

**Capabilities**:
- Text messages: Routed through conversation pipeline
- Voice notes: Downloaded OGG -> Deepgram transcription -> review UI or auto-process
- Photos: Downloaded -> GPT-4o-mini vision description -> conversation pipeline
- Commands: `/voice [review|handsfree]`, `/voicereply [on|off]`
- Outbound: Proactive messages, image delivery (polls for generation completion)

Long polling with 30s timeout, updates every 1s.

## Image Storage (`image/storage_provider.py`)

Abstraction layer for persistent image hosting. Generated images are uploaded after creation so URLs don't expire.

**Provider selection**: Set `IMAGE_STORAGE_PROVIDER` env var:

| Provider | Env Value | Config Required | Notes |
|----------|-----------|-----------------|-------|
| **Cloudinary** | `cloudinary` (default) | `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET` | Free tier available. Images stored at `companion/{task_id}`. |
| **AWS S3** | `s3` | `AWS_S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Set `AWS_S3_REGION` (default: us-east-1). Supports S3-compatible services via `AWS_S3_ENDPOINT_URL`. |

Both providers implement the same `ImageStorageProvider` interface (`upload(image_url, key) -> url`). If the provider isn't configured, the system gracefully degrades to using ephemeral RunComfy URLs.

### Cloudinary Setup

1. Sign up at [cloudinary.com](https://cloudinary.com/)
2. Go to Dashboard > API Keys
3. Set in `.env`:
   ```
   IMAGE_STORAGE_PROVIDER=cloudinary
   CLOUDINARY_CLOUD_NAME=your_cloud_name
   CLOUDINARY_API_KEY=your_api_key
   CLOUDINARY_API_SECRET=your_api_secret
   ```

### AWS S3 Setup

1. Sign in to the [AWS Console](https://console.aws.amazon.com/)
2. Go to **S3** and create a bucket:
   - Choose a region (e.g., `us-east-1`)
   - Uncheck "Block all public access" (images need to be publicly readable)
   - Confirm the public access warning
3. Go to **IAM > Users** and create a new user (or use an existing one):
   - Attach the `AmazonS3FullAccess` policy (or a scoped policy for your bucket)
   - Go to **Security credentials** tab > **Create access key**
   - Choose "Application running outside AWS"
   - Copy the **Access key ID** and **Secret access key**
4. Set in `.env`:
   ```
   IMAGE_STORAGE_PROVIDER=s3
   AWS_S3_BUCKET=your-bucket-name
   AWS_S3_REGION=us-east-1
   AWS_ACCESS_KEY_ID=AKIAxxxxxxxxxxxxxxxx
   AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

For **S3-compatible services** (MinIO, DigitalOcean Spaces, Backblaze B2):
```
AWS_S3_ENDPOINT_URL=https://nyc3.digitaloceanspaces.com
```

## n8n (Workflow Automation)

Connected via Caddy proxy. Used by ToolRouter for calendar webhook operations.

---

## Voice System

**Source**: `src/voice/`

### Speech-to-Text (STT)

Provider: **Deepgram Nova-2** via REST API. Env: `DEEPGRAM_API_KEY`.

### Text-to-Speech (TTS)

| Provider | Quality | Cost | Notes |
|----------|---------|------|-------|
| ElevenLabs | High | Paid | Customizable voice_id, preserves `(sighs)` and `(laughs)` |
| Edge TTS | Good | Free | Microsoft voices, zero-cost fallback |

Text cleaning strips `*actions*` and markdown but preserves parenthetical expressions for ElevenLabs.

### Voice State (`voice/voice_state.py`)

Redis-backed per-user state: voice processing mode (review vs handsfree), voice_reply enabled/disabled per chat_id.

---

## Image Generation

**Source**: `src/image/`

### Providers

| Provider | API | Cost | Best For |
|----------|-----|------|----------|
| NanoBanana | Google Gemini | Free | General images, mood variations from base avatar |
| RunComfy | ComfyUI workflows | Paid | LoRA-tuned models, identity-consistent images |

### Per-Companion Configuration

```yaml
# In persona.yaml
image:
  provider: nanobanana           # default
  # provider: runcomfy
  # runcomfy_deployment_id: "abc123"   # If set, RunComfy used automatically
  content_rules: gemini          # 'gemini' or 'unrestricted'
```

If a companion has a `runcomfy_deployment_id`, RunComfy is used regardless of the `provider` setting.

### Content Rules

| Rule | Provider | Allowed |
|------|----------|---------|
| `gemini` | NanoBanana | No NSFW, limited violence, no real person likenesses |
| `unrestricted` | RunComfy | Depends on ComfyUI workflow and LoRA training |

### Image Delivery

1. Intent detected in pipeline Stage 5
2. Celery task queued for generation
3. Generated image uploaded to Cloudinary
4. Redis pub/sub notification
5. WebSocket or Telegram delivery to user
