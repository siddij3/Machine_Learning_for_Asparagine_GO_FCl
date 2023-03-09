import boto3

def s3_bucket():
    return boto3.resource(
        service_name='s3',
        region_name='us-east-2',
        aws_access_key_id='AKIAS43V7ZXDBIM35FEP',
        aws_secret_access_key='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'
    )
    

def save_to_bucket(s3, filename):

    s3.meta.client.upload_file('WQM_NNs.zip', 'wqm-ml-models', 'WQM_NNs.zip')


# def load_from_bucket():

