# Zoom OAuth Integration

Attendee's Zoom OAuth integration allows your bots to record Zoom meetings without asking permission from the host. This is accomplished by storing your users' Zoom OAuth credentials in Attendee, which are then used to generate local recording tokens for meetings hosted by those users.

The guide below walks through how to set up Zoom OAuth integration in your app.

## How it works

When a user authorizes your Zoom app through OAuth:
1. Your application sends the authorization code to Attendee
2. Attendee exchanges it for OAuth credentials and stores them in Attendee
3. When your bot joins a meeting hosted by that user, Attendee generates a join token using the stored credentials
4. Due to the join token, the bot has its recording permissions pre-approved, so it does not need to ask permission from the host.

## Create a Zoom App

You'll need to create a Zoom OAuth App that your users will authorize. We recommend creating separate apps for development and production.

1. Go to the [Zoom App Marketplace](https://marketplace.zoom.us/) and click "Develop" → "Build App"
2. Choose "General App" as the app type
3. Fill in your app information (name, description, etc.)
4. In the "Scopes" section, add the required scopes:
   - `meeting:read` - to read meeting information
   - `user:read` - to read user information
5. In the "Redirect URL for OAuth" section, add your application's OAuth callback URL
6. In the "Event Subscriptions" section, you'll configure webhooks (see below)
7. Note your **Client ID** and **Client Secret**—you'll need these in the next step

## Register your Zoom App with Attendee

Once you've created your Zoom app, you need to register it with Attendee:

1. Make a POST request to the Attendee API to create a Zoom OAuth App:
   ```
   POST /api/v1/zoom_oauth_apps
   ```
   Include your Zoom app's `client_id` and `client_secret` in the request body.

2. Save the `zoom_oauth_app_id` from the response—you'll need this when creating Zoom OAuth Connections.

## Configure Zoom App Webhooks

Your Zoom app needs to send webhook events to Attendee so that Attendee knows when meetings are starting and can manage OAuth connections.

1. In your Zoom App settings, go to the "Event Subscriptions" section
2. Add the Attendee webhook endpoint as your Event notification endpoint URL:
   ```
   https://api.attendee.dev/api/v1/zoom_oauth/webhooks
   ```
   (Replace with your Attendee instance URL if you're self-hosting)
3. Subscribe to these event types:
   - `meeting.started`
   - `meeting.ended`
   - `endpoint.url_validation` (required by Zoom)
4. Save your changes

## Add OAuth Flow Logic to Your Application

You'll need to add code to handle the OAuth flow for users to authorize your Zoom app. Here's the typical flow:

1. **Add an auth endpoint** that redirects users to Zoom's OAuth authorization URL:
   ```
   https://zoom.us/oauth/authorize?response_type=code&client_id={YOUR_CLIENT_ID}&redirect_uri={YOUR_REDIRECT_URI}
   ```

2. **Add a callback endpoint** that handles the OAuth callback from Zoom. When the user authorizes your app, Zoom will redirect to this endpoint with an authorization code.

3. **In your callback endpoint**, after receiving the authorization code from Zoom, make a POST request to Attendee to create a Zoom OAuth Connection:
   ```
   POST /api/v1/zoom_oauth_connections
   ```
   Include in the request body:
   - `zoom_oauth_app_id` - The ID of the Zoom OAuth App you registered with Attendee
   - `authorization_code` - The authorization code from Zoom
   - `redirect_uri` - The redirect URI used in the OAuth flow
   - `deduplication_key` - (Optional but recommended) A unique identifier for this user (e.g., their email or internal user ID) to prevent duplicate connections

4. **Save the connection**: Attendee will exchange the authorization code for OAuth credentials and return a Zoom OAuth Connection object. Save this connection object (particularly the `zoom_oauth_connection_id`) to your database associated with the user.

## Using Zoom OAuth Connections with Bots

Once a Zoom OAuth Connection is created, any bots joining meetings hosted by that user will automatically use join tokens (no permission prompt required).

When creating a bot for a Zoom meeting, you can optionally include the `zoom_oauth_connection_id` parameter in your bot creation request:

```
POST /api/v1/bots
{
  "meeting_url": "https://zoom.us/j/123456789",
  "zoom_oauth_connection_id": "zoauth_abc123",
  ...
}
```

If you don't specify a `zoom_oauth_connection_id`, Attendee will automatically try to find an appropriate connection based on the meeting host. However, explicitly providing the connection ID ensures the correct credentials are used.

## Handling Disconnected Connections

Zoom OAuth Connections can become disconnected if:
- The user revokes access to your Zoom app
- The OAuth credentials expire and cannot be refreshed
- The user's Zoom account is deleted

To monitor connection health, you can:
1. Periodically check the `state` field of your Zoom OAuth Connections via a GET request to `/api/v1/zoom_oauth_connections/{connection_id}`
2. Set up webhooks to be notified when a connection becomes disconnected (if this webhook type is available)

When a connection becomes disconnected, bots will fall back to the standard join flow (with permission prompts).

## FAQ

### Do I need to create a Zoom OAuth Connection for every user?

Only for users who will be hosting meetings that your bots need to join without permission prompts. If your bots are joining meetings where the host hasn't authorized your app, the bots will use the standard join flow.

### Can I use the same Zoom App for multiple Attendee projects?

While technically possible, we recommend creating separate Zoom apps for different environments (development, staging, production) and corresponding Attendee projects for better isolation and security.

### What happens if the Zoom OAuth credentials expire?

Attendee automatically refreshes OAuth credentials when they expire. If refresh fails (e.g., user revoked access), the connection will move to a `disconnected` state.