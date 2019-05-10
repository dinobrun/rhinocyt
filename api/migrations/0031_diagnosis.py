# Generated by Django 2.2 on 2019-05-09 08:39

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0030_delete_diagnosis'),
    ]

    operations = [
        migrations.CreateModel(
            name='Diagnosis',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('diagnosis_description', models.CharField(blank=True, default='', max_length=100)),
                ('anamnesis', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='diagnosis', to='api.Anamnesis')),
                ('cell_extraction', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='diagnosis', to='api.CellExtraction')),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='diagnosis', to='api.Patient')),
            ],
            options={
                'abstract': False,
            },
        ),
    ]
