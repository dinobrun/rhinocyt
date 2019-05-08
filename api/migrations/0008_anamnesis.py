# Generated by Django 2.2 on 2019-05-06 09:08

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0007_auto_20190214_2010'),
    ]

    operations = [
        migrations.CreateModel(
            name='Anamnesis',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('allergy', models.CharField(choices=[('AL', 'Alimenti'), ('IN', 'Inalanti')], max_length=2)),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='patient', to='api.Patient')),
            ],
            options={
                'abstract': False,
            },
        ),
    ]
