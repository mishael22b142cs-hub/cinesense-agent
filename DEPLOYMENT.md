# Deploying CineSense Agent to AWS (App Runner + ECR)

This deploys the **FastAPI** interface (`api.py`), as a second, independent deployment
alongside the existing Streamlit Community Cloud app — it does not touch or replace that
live demo. Run every command below yourself, in your own terminal, after `aws configure`
with your own credentials. I never see or touch your AWS keys — fill in the real values
anywhere you see `<...>`.

**Cost warning:** App Runner bills by the hour while running, even idle (roughly a few
dollars a month at the smallest size). Set a [billing alarm](https://console.aws.amazon.com/billing/home#/budgets)
first, and run the cleanup commands at the bottom when you're done demoing it.

## 0. Prerequisites

- [Install the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).
- Create an IAM user for this (not your root account): AWS Console → IAM → Users →
  Create user → attach `AmazonEC2ContainerRegistryFullAccess` and `AWSAppRunnerFullAccess`
  → create an access key (CLI use case).
- `aws configure` with that user's keys and a region (e.g. `ap-south-1` for Mumbai).
- This branch (`vector-search-rag`) merged into `main` on GitHub once you've tested it
  (see the top-level note in README.md about testing via a second Streamlit app first).

## 1. Create the ECR repository (one-time)

```bash
aws ecr create-repository --repository-name cinesense-agent --region <your-region>
```

Note the `repositoryUri`, e.g. `<account-id>.dkr.ecr.<region>.amazonaws.com/cinesense-agent`.

## 2. Add GitHub Actions secrets

GitHub repo → Settings → Secrets and variables → Actions → New repository secret:
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`.

`.github/workflows/deploy.yml` builds the Docker image and pushes it to ECR on every
push to `main`.

## 3. Create an IAM role App Runner can use to pull from ECR

```bash
aws iam create-role --role-name AppRunnerECRAccessRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "build.apprunner.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

aws iam attach-role-policy --role-name AppRunnerECRAccessRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
```

## 4. Create the App Runner service

```bash
aws apprunner create-service \
  --service-name cinesense-agent \
  --region <your-region> \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "<account-id>.dkr.ecr.<region>.amazonaws.com/cinesense-agent:latest",
      "ImageRepositoryType": "ECR",
      "ImageConfiguration": {
        "Port": "8000",
        "RuntimeEnvironmentVariables": {"GROQ_API_KEY": "<your-real-groq-key>"}
      }
    },
    "AutoDeploymentsEnabled": true,
    "AuthenticationConfiguration": {"AccessRoleArn": "arn:aws:iam::<account-id>:role/AppRunnerECRAccessRole"}
  }' \
  --instance-configuration '{"Cpu": "0.25 vCPU", "Memory": "0.5 GB"}'
```

This prints a `ServiceUrl`. After a few minutes (status `RUNNING`):

```bash
curl -X POST https://<service-url>/ask -H "Content-Type: application/json" \
  -d '{"question": "Recommend 3 movies like The Dark Knight."}'
```

`AutoDeploymentsEnabled: true` means every future push to `main` auto-redeploys —
real CI/CD, not a one-off deploy.

## Cleanup (stop paying for it)

```bash
aws apprunner list-services --region <your-region>   # get the ServiceArn
aws apprunner delete-service --service-arn <service-arn> --region <your-region>
aws ecr delete-repository --repository-name cinesense-agent --region <your-region> --force
```
