from django.conf.urls import patterns, url


urlpatterns = patterns(
    'images',
    url(r'^compute/$', 'views.compute')
)
