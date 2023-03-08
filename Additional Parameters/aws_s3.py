import boto3

def s3_bucket():
    return boto3.resource(
        service_name='s3',
        region_name='us-east-2',
        aws_access_key_id='AKIAS43V7ZXDESJNGQUB',
        aws_secret_access_key='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'
    )
    

def save_to_bucket(s3, filename):

    s3.meta.client.upload_file('ML_Models.zip', 'wqm-ml-models', 'ML_Models.zip')


# def load_from_bucket():

