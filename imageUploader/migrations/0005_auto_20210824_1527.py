# Generated by Django 3.1.4 on 2021-08-24 09:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('imageUploader', '0004_auto_20210824_1517'),
    ]

    operations = [
        migrations.AlterField(
            model_name='image',
            name='image',
            field=models.ImageField(upload_to='images/'),
        ),
    ]
