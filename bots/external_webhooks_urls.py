from django.urls import path

from . import external_webhooks_views

urlpatterns = [
    path(
        "stripe",
        external_webhooks_views.ExternalWebhookStripeView.as_view(),
        name="external-webhook-stripe",
    ),
    path(
        "zoom/oauth_apps/<str:object_id>",
        external_webhooks_views.ExternalWebhookZoomOAuthAppView.as_view(),
        name="external-webhook-zoom-oauth-app",
    ),
]
