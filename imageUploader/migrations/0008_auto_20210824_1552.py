# Generated by Django 3.1.4 on 2021-08-24 10:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('imageUploader', '0007_auto_20210824_1545'),
    ]

    operations = [
        migrations.CreateModel(
            name='Image',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='uploads/')),
            ],
        ),
        migrations.DeleteModel(
            name='Hotel',
        ),
    ]
