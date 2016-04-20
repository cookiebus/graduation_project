# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import jsonfield.fields


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Image',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('image', models.ImageField(upload_to=b'media/images/problems', blank=True)),
                ('model_3D', models.ImageField(upload_to=b'media/3D_models/problems', blank=True)),
                ('kp', jsonfield.fields.JSONField(null=True, blank=True)),
                ('des', jsonfield.fields.JSONField(null=True, blank=True)),
                ('create_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'ordering': ('id',),
            },
            bases=(models.Model,),
        ),
    ]
