from django.conf.urls import patterns, url


urlpatterns = patterns(
    'api',
    url(r'^upload/$', 'views.upload_image')
)
