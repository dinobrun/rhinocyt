# Generated by Django 2.1.5 on 2019-01-19 11:54

from django.conf import settings
import django.core.validators
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Cell',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='cells/')),
                ('validated', models.BooleanField(blank=True, default=False)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='CellCategory',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('classnum', models.SmallIntegerField()),
                ('name', models.CharField(max_length=30)),
            ],
            options={
                'verbose_name_plural': 'cell categories',
            },
        ),
        migrations.CreateModel(
            name='CellExtraction',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('extraction_date', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='City',
            fields=[
                ('code', models.CharField(max_length=6, primary_key=True, serialize=False, validators=[django.core.validators.MinLengthValidator(6, 'Length has to be 6.')])),
                ('name', models.CharField(max_length=100)),
                ('province_code', models.CharField(max_length=2, validators=[django.core.validators.MinLengthValidator(2, 'Length has to be 2.')])),
            ],
            options={
                'verbose_name_plural': 'cities',
            },
        ),
        migrations.CreateModel(
            name='Doctor',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Patient',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('first_name', models.CharField(blank=True, default='', max_length=25)),
                ('last_name', models.CharField(blank=True, default='', max_length=25)),
                ('fiscal_code', models.CharField(blank=True, default='', max_length=16, validators=[django.core.validators.MinLengthValidator(16, message='Length has to be 16.')])),
                ('birthdate', models.DateField(blank=True, null=True)),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('birth_city', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='birth_city', to='api.City')),
                ('doctor', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='patients', to='api.Doctor')),
                ('residence_city', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='residence_city', to='api.City')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Slide',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='slides/')),
                ('cell_extraction_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='slide_id', to='api.CellExtraction')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.AddField(
            model_name='cell',
            name='cell_category_id',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='cells', to='api.CellCategory'),
        ),
        migrations.AddField(
            model_name='cell',
            name='patient_id',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='cells', to='api.Patient'),
        ),
        migrations.AddField(
            model_name='cell',
            name='slide_id',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='cells', to='api.Slide'),
        ),
    ]
