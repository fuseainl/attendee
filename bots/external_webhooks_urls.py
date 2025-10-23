from django.urls import path

from . import external_webhooks_views

app_name = "bots"

urlpatterns = [
    path(
        "stripe",
        external_webhooks_views.ExternalWebhookStripeView.as_view(),
        name="external-webhook-stripe",
    ),
    path(
        "google_calendar",
        external_webhooks_views.ExternalWebhookGoogleCalendarView.as_view(),
        name="external-webhook-google-calendar",
    ),
    path(
        "zoom/oauth_apps/<str:object_id>",
        external_webhooks_views.ExternalWebhookZoomOAuthAppView.as_view(),
        name="external-webhook-zoom-oauth-app",
    ),
]
