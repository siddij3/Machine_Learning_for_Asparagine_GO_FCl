import boto3
import shutil

def s3_bucket():
    return boto3.resource(
        service_name='s3',
        region_name='us-east-2',
        aws_access_key_id='AKIAS43V7ZXDIMVQ4BPY',
        aws_secret_access_key='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'
    )
    

def save_to_bucket(s3, filename):
    s3.meta.client.upload_file(filename, get_bucket_name(), filename)
    print("Uploaded to AWS")


def load_from_bucket(s3, path):
    for bucket in s3.buckets.all():
        if ("wqm" in bucket.name):
            for obj in bucket.objects.all():
                print(obj.key)

                with open(obj.key, 'wb') as data:
                    if ("testing" in obj.key):
                        s3.meta.client.download_fileobj(bucket.name, obj.key, data)
                        shutil.unpack_archive(obj.key, path)

def get_bucket_name():
    return 'wqm-ml-models'

