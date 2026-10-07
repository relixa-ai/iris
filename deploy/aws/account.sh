#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env

say() { printf '[account] %s\n' "$*"; }

if aws s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1; then
  say "bucket s3://$BUCKET exists"
else
  if [ "$REGION" = us-east-1 ]; then
    aws s3api create-bucket --bucket "$BUCKET" >/dev/null
  else
    aws s3api create-bucket --bucket "$BUCKET" \
      --create-bucket-configuration LocationConstraint="$REGION" >/dev/null
  fi
  aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  say "bucket s3://$BUCKET created (private)"
fi

if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --tags Key=Project,Value="$CLUSTER" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
      "Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  say "role $ROLE created"
fi
aws iam put-role-policy --role-name "$ROLE" --policy-name "$CLUSTER-bucket" \
  --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[
    {\"Effect\":\"Allow\",\"Action\":[\"s3:ListBucket\",\"s3:GetBucketLocation\",
      \"s3:ListBucketMultipartUploads\"],\"Resource\":\"arn:aws:s3:::$BUCKET\"},
    {\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:DeleteObject\",
      \"s3:AbortMultipartUpload\",\"s3:ListMultipartUploadParts\"],
     \"Resource\":\"arn:aws:s3:::$BUCKET/*\"}]}"
if ! aws iam get-instance-profile --instance-profile-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$ROLE" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE"
  say "instance profile $ROLE created (IAM takes a few seconds to see it)"
  sleep 10
fi
say "role $ROLE: s3://$BUCKET only"

if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" >/dev/null 2>&1; then
  mkdir -p "$(dirname "$KEY_FILE")"
  aws ec2 create-key-pair --key-name "$KEY_NAME" --key-type ed25519 \
    --tag-specifications "ResourceType=key-pair,Tags=[{Key=Project,Value=$CLUSTER}]" \
    --query KeyMaterial --output text > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
  say "key pair $KEY_NAME created: $KEY_FILE"
elif [ ! -f "$KEY_FILE" ]; then
  echo "key pair $KEY_NAME exists in AWS but $KEY_FILE does not: delete the pair" \
       "(aws ec2 delete-key-pair --key-name $KEY_NAME) and run again" >&2
  exit 1
else
  say "key pair $KEY_NAME: $KEY_FILE"
fi

VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' \
      --output text)
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values="$SG_NAME" \
     Name=vpc-id,Values="$VPC" --query 'SecurityGroups[0].GroupId' --output text)
if [ "$SG" = None ]; then
  SG=$(aws ec2 create-security-group --group-name "$SG_NAME" --vpc-id "$VPC" \
       --description "iris Spark cluster: nodes to each other, SSH from the owner" \
       --tag-specifications "ResourceType=security-group,Tags=[{Key=Project,Value=$CLUSTER}]" \
       --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --group-id "$SG" \
    --ip-permissions "IpProtocol=-1,UserIdGroupPairs=[{GroupId=$SG}]" >/dev/null
  say "security group $SG_NAME ($SG) created"
fi
MY_IP=$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')
aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 \
  --cidr "$MY_IP/32" >/dev/null 2>&1 || true
say "security group $SG: all traffic between nodes; SSH from $MY_IP"
say "done"
