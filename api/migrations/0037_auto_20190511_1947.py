# Generated by Django 2.2 on 2019-05-11 17:47

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0036_auto_20190511_1931'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='anamnesis',
            name='prick_test',
        ),
        migrations.DeleteModel(
            name='PrickTest',
        ),
    ]
