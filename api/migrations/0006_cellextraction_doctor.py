# Generated by Django 2.1.5 on 2019-02-12 00:43

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0005_auto_20190119_1211'),
    ]

    operations = [
        migrations.AddField(
            model_name='cellextraction',
            name='doctor',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, to='api.Doctor'),
            preserve_default=False,
        ),
    ]