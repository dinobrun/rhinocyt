# Generated by Django 2.2 on 2019-05-10 15:10

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0034_auto_20190510_1709'),
    ]

    operations = [
        migrations.CreateModel(
            name='DiagnosisExtraction',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('diagnosis_date', models.DateTimeField(auto_now_add=True)),
                ('anamnesis', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='diagnosis_extraction', to='api.Anamnesis')),
                ('cell_extraction', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='diagnosis_extraction', to='api.CellExtraction')),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='diagnosis_extraction', to='api.Patient')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Diagnosis',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=30)),
                ('information', models.CharField(max_length=300)),
                ('diagnosis_extraction', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='diagnosis', to='api.DiagnosisExtraction')),
            ],
            options={
                'abstract': False,
            },
        ),
    ]