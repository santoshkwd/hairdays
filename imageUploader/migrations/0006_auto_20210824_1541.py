# Generated by Django 3.1.4 on 2021-08-24 10:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('imageUploader', '0005_auto_20210824_1527'),
    ]

    operations = [
        migrations.AlterField(
            model_name='image',
            name='image',
            field=models.ImageField(upload_to='FGHJKLHGYUHIGUI/'),
        ),
    ]
