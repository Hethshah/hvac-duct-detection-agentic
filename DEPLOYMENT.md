# Deployment Strategy — HVAC Duct Detection Pipeline

Three deployment scenarios: local development, small-scale cloud (15–20 req/day), and mass-scale cloud (500–1,000 req/day) using AWS AgentCore.

---

## Cost Assumptions (all three scenarios)

The Anthropic API cost dominates every scenario. The estimate below is for **one pipeline run on a single-page drawing with no retries**.

| Agent | Model | Typical tokens (in / out) | Cost |
|---|---|---|---|
| Ingestion | claude-haiku-4-5 | 800 / 300 | ~$0.002 |
| Vision | claude-opus-4-7 | 13,000 / 3,200 (4 quadrants) | ~$0.43 |
| Measurement | claude-sonnet-4-6 | 3,000 / 500 | ~$0.017 |
| Review | claude-sonnet-4-6 | 2,000 / 300 | ~$0.010 |
| **Total (1 page, 0 retries)** | | | **~$0.46** |
| **Total (1 page, 1 retry)** | | +1 vision+measure+review cycle | **~$0.90** |
| **Total (3-page drawing, 0 retries)** | | 3× vision cost | **~$1.35** |

> Vision is the dominant cost because claude-opus-4-7 processes four ~1,300 × 1,650 px quadrant images per page at $15/MTok input and $75/MTok output. Switching to claude-sonnet-4-6 for vision would cut per-run cost by ~60% with some detection quality trade-off.

---

## Scenario 1 — Local (Current)

### How it works today

```
Developer Machine
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  $ python hvac-duct-detection/scripts/run_pipeline.py   │
│           --pdf "sample input/input.pdf"                 │
│                                                          │
│  Orchestrator ──► Ingestion ──► Vision ──► Measurement  │
│                       │            │                     │
│                    PyMuPDF      Anthropic API            │
│                   (local)      (remote call)             │
│                                                          │
│  Output: hvac-duct-detection/outputs/<session_id>/       │
│  Log:    hvac-duct-detection/runs/registry.csv           │
└──────────────────────────────────────────────────────────┘
```

### Setup requirements

- Python 3.10+, `.venv`, `requirements.txt`
- `ANTHROPIC_API_KEY` in `.env`
- No cloud accounts needed

### Cost per run

| Component | Cost |
|---|---|
| Anthropic API (1 page, no retry) | ~$0.46 |
| Anthropic API (1 page, 1 retry) | ~$0.90 |
| Infrastructure | $0.00 (local machine) |
| **Effective per run** | **~$0.46 – $0.90** |

### Limitations

- Runs one job at a time (no concurrency)
- No job queue — caller blocks until pipeline finishes (~2–8 min per page)
- Results stay on local disk; no remote access
- Machine must stay on for the full duration of every run

---

## Scenario 2 — Small Scale: 15–20 Requests / Day on AWS

### Target load

| Metric | Value |
|---|---|
| Requests per day | 15–20 |
| Requests per month | ~500 |
| Concurrency (peak) | 3–5 simultaneous jobs |
| Expected pipeline duration | 3–10 min per job |

### Architecture

```
                         ┌─────────────────┐
  Client (browser / API) │  API Gateway    │  POST /v1/jobs
  ──────────────────────►│  (HTTP API)     │  GET  /v1/jobs/{id}
                         └────────┬────────┘  GET  /v1/jobs/{id}/download
                                  │
                         ┌────────▼────────┐
                         │   Lambda        │  Validates request,
                         │   (dispatcher)  │  generates session_id,
                         └────────┬────────┘  creates pre-signed S3 URL
                                  │
                    ┌─────────────▼──────────────┐
                    │            SQS             │  Job queue
                    │  (StandardQueue, vis=15min)│  (buffers bursts)
                    └─────────────┬──────────────┘
                                  │ triggers
                    ┌─────────────▼──────────────┐
                    │       ECS Fargate           │  1 vCPU · 2 GB RAM
                    │   (pipeline container)      │  Runs full Python pipeline
                    │                             │  Calls Anthropic API directly
                    └──────┬──────────────┬───────┘
                           │              │
               ┌───────────▼──┐    ┌──────▼──────────┐
               │  S3 Bucket   │    │   DynamoDB       │
               │  ├ inputs/   │    │   jobs table     │
               │  └ outputs/  │    │   (status, paths)│
               └──────────────┘    └──────────────────┘
                                          │
                               ┌──────────▼──────────┐
                               │  SNS → Email / Slack │
                               │  (job complete alert)│
                               └─────────────────────┘
```

### Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| Compute | ECS Fargate on-demand | No idle cost; scales to 0; supports 15+ min jobs (Lambda max = 15 min) |
| Queue | SQS Standard | Absorbs bursts; decouples submission from execution; built-in visibility timeout |
| Storage | S3 | Durable, cheap, pre-signed URLs for direct client upload/download |
| State | DynamoDB | Serverless, low-cost at this volume; stores job status + all registry fields |
| Concurrency cap | SQS → Fargate max tasks = 5 | Prevents runaway API spend during bursts |

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/jobs` | Submit a job — returns `session_id` and a pre-signed S3 URL to upload the PDF |
| `GET` | `/v1/jobs/{session_id}` | Poll job status: `queued`, `running`, `complete`, `failed` |
| `GET` | `/v1/jobs/{session_id}/result` | Download results as a ZIP (annotated PDF + JSON + CSV) |
| `GET` | `/v1/jobs` | List recent jobs with optional `?status=complete&limit=50` filter |

**Example: submit a job**

```bash
# 1. Create the job
curl -X POST https://api.example.com/v1/jobs \
  -H "x-api-key: <your-key>" \
  -H "Content-Type: application/json" \
  -d '{"filename": "floor-plan.pdf", "pages": "1-3", "confidence": 0.85}'

# Response
{
  "session_id": "20260428_143200_a1b2c3",
  "upload_url": "https://hvac-inputs.s3.amazonaws.com/...",
  "status_url": "https://api.example.com/v1/jobs/20260428_143200_a1b2c3"
}

# 2. Upload the PDF directly to S3 (no server round-trip)
curl -X PUT "<upload_url>" --upload-file floor-plan.pdf

# 3. Poll for completion
curl https://api.example.com/v1/jobs/20260428_143200_a1b2c3 \
  -H "x-api-key: <your-key>"

# Response when done
{
  "session_id": "20260428_143200_a1b2c3",
  "status": "complete",
  "segments_detected": 15,
  "segments_labelled": 15,
  "review_score": 0.964,
  "retries": 0,
  "result_url": "https://api.example.com/v1/jobs/20260428_143200_a1b2c3/result"
}
```

### Cost per run

| Component | Monthly (500 runs) | Per run |
|---|---|---|
| ECS Fargate (1 vCPU, 2 GB, ~8 min avg) | ~$5 | ~$0.010 |
| S3 (storage + requests) | ~$5 | ~$0.010 |
| DynamoDB (on-demand) | ~$2 | ~$0.004 |
| SQS + Lambda + API Gateway | ~$2 | ~$0.004 |
| CloudWatch logs/metrics | ~$3 | ~$0.006 |
| **AWS infrastructure total** | **~$17/month** | **~$0.034** |
| **Anthropic API (avg $0.60/run)** | **~$300/month** | **~$0.60** |
| **Grand total** | **~$317/month** | **~$0.63** |

> The Anthropic API (~95% of cost) completely dominates. AWS infrastructure at this scale is almost negligible.

---

## Scenario 3 — Mass Scale: 500–1,000 Requests / Day on AWS with AgentCore

### Target load

| Metric | Value |
|---|---|
| Requests per day | 500–1,000 |
| Requests per month | ~22,500 (at 750/day avg) |
| Peak concurrency | 50–80 simultaneous jobs |
| SLA target | Results within 15 min of submission |

### Architecture

```
                   ┌───────────────────────────────┐
                   │         CloudFront CDN         │
                   │  (serves output files globally)│
                   └──────────────┬────────────────┘
                                  │
┌──────────┐      ┌───────────────▼────────────────┐
│  Clients │─────►│      API Gateway (REST)         │
│(web/API) │      │  + WAF + Usage Plans + API Keys │
└──────────┘      └──────────┬──────────────────────┘
                             │
              ┌──────────────▼──────────────────┐
              │          SQS FIFO Queue          │
              │   (per-tenant message groups;    │
              │    max receive = 80 concurrent)  │
              └──────────────┬──────────────────┘
                             │
    ┌────────────────────────▼────────────────────────────┐
    │           AWS Step Functions (Standard)              │
    │                                                      │
    │  ┌──────────┐  ┌────────┐  ┌───────────┐  ┌──────┐│
    │  │Ingestion │─►│Vision  │─►│Measurement│─►│Annot.││
    │  │ Agent    │  │ Agent  │  │  Agent    │  │Agent ││
    │  └──────────┘  └───┬────┘  └───────────┘  └──┬───┘│
    │                    │  score < threshold?       │    │
    │                    │◄──────────────────────────┘    │
    │                    │   Review Agent (retry loop)    │
    │                    │   max 3 retries                │
    │                    │   2^n second back-off          │
    └────────────────────┼────────────────────────────────┘
                         │
    ┌────────────────────▼────────────────────────────────┐
    │         Amazon Bedrock AgentCore Runtime             │
    │                                                      │
    │  ┌──────────────────────────────────────────────┐   │
    │  │  Each agent runs as an AgentCore managed     │   │
    │  │  agent — fully containerized, auto-scaled,   │   │
    │  │  with built-in:                              │   │
    │  │   • Tool execution environment               │   │
    │  │   • Session / state management               │   │
    │  │   • Observability (traces + metrics)         │   │
    │  │   • Automatic retries on transient errors    │   │
    │  └──────────────────────────────────────────────┘   │
    │                                                      │
    │  Model inference ──► Amazon Bedrock                  │
    │  (Claude claude-opus-4-7 / sonnet / haiku via        │
    │   Bedrock on-demand + batch inference profiles)      │
    └──────────────────────┬──────────────────────────────┘
                           │
          ┌────────────────┼────────────────────┐
          │                │                    │
    ┌─────▼──────┐  ┌──────▼──────┐  ┌─────────▼────────┐
    │  S3 Bucket │  │  DynamoDB   │  │  ElastiCache     │
    │  inputs/   │  │  jobs table │  │  (Redis)         │
    │  outputs/  │  │  registry   │  │  dedup cache for │
    │  (+ S3     │  │             │  │  identical pages │
    │  Lifecycle)│  │             │  └──────────────────┘
    └────────────┘  └─────────────┘
          │
    ┌─────▼──────────────────────────────┐
    │  CloudWatch + AWS X-Ray            │
    │  • Per-agent latency dashboards    │
    │  • Cost-per-run metric             │
    │  • Retry rate alarms               │
    │  • Dead-letter queue monitoring    │
    └────────────────────────────────────┘
```

### Why AgentCore at this scale

| Without AgentCore | With AgentCore |
|---|---|
| You manage container lifecycle, scaling, health checks | Fully managed — AgentCore handles runtime scaling |
| Tool execution runs inside your own ECS tasks | Isolated, sandboxed tool execution per agent invocation |
| Manual observability wiring | Native traces, metrics, and logs per agent step |
| You implement retry/fault logic per agent | Built-in retry policies and circuit breakers |
| Session state managed by your DynamoDB code | AgentCore Memory manages short-term and long-term context |

### Key design decisions at scale

| Decision | Choice | Reason |
|---|---|---|
| Model inference | Amazon Bedrock (not direct Anthropic API) | Rate limit management, batch inference (50% cheaper), centralized IAM auth |
| Batch inference | Bedrock Batch for non-urgent jobs | 50% token cost reduction for jobs not needing real-time response |
| Workflow | Step Functions Standard | Durable execution history, visual debugging, native retry with exponential back-off |
| Caching | ElastiCache Redis | Cache vision results for duplicate page hashes — avoids redundant API calls for the same drawing |
| Storage lifecycle | S3 Intelligent-Tiering | Auto-moves outputs to cheaper storage tiers after 30 days |
| Concurrency control | SQS FIFO + Step Functions concurrency limit | Caps parallel Bedrock calls to stay within model throughput limits |
| Observability | X-Ray traces + CloudWatch EMF | Per-segment latency, per-agent token spend, retry rate — all queryable |

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/jobs` | Submit a job — body includes presigned S3 URL request |
| `GET` | `/v1/jobs/{session_id}` | Job status + progress (which agent is currently running) |
| `GET` | `/v1/jobs/{session_id}/result` | Download annotated PDF, JSON, CSV as ZIP |
| `DELETE` | `/v1/jobs/{session_id}` | Cancel a queued or running job |
| `GET` | `/v1/jobs` | List jobs with filtering (`?status=`, `?from=`, `?limit=`) |
| `POST` | `/v1/jobs/batch` | Submit up to 100 jobs in one call (async, Bedrock batch mode) |
| `GET` | `/v1/usage` | Token spend, run count, retry rate for the current billing period |
| `GET` | `/v1/health` | Service health + model availability status |

**Example: batch submit**

```bash
curl -X POST https://api.example.com/v1/jobs/batch \
  -H "x-api-key: <your-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "jobs": [
      {"filename": "floor-M101.pdf", "pages": "1"},
      {"filename": "floor-M102.pdf", "pages": "1-4"},
      {"filename": "floor-M103.pdf"}
    ],
    "confidence": 0.85,
    "mode": "batch"
  }'

# Response
{
  "batch_id": "batch_20260428_1500_xyz",
  "jobs": [
    {"session_id": "20260428_150001_aa1", "upload_url": "..."},
    {"session_id": "20260428_150001_bb2", "upload_url": "..."},
    {"session_id": "20260428_150001_cc3", "upload_url": "..."}
  ],
  "estimated_completion_minutes": 45
}
```

### Cost per run at scale

**On-demand pricing (time-sensitive jobs)**

| Component | Monthly (22,500 runs) | Per run |
|---|---|---|
| Amazon Bedrock — claude-opus-4-7 vision | ~$9,675 | ~$0.43 |
| Amazon Bedrock — claude-sonnet-4-6 (measure + review) | ~$608 | ~$0.027 |
| Amazon Bedrock — claude-haiku-4-5 (ingestion) | ~$45 | ~$0.002 |
| AgentCore Runtime (est. $0.03/invocation) | ~$675 | ~$0.030 |
| Step Functions (10 state transitions/run) | ~$6 | ~$0.0003 |
| ECS Fargate (preprocessing + orchestrator) | ~$225 | ~$0.010 |
| S3 (storage + data transfer) | ~$150 | ~$0.007 |
| DynamoDB (on-demand) | ~$75 | ~$0.003 |
| ElastiCache Redis (cache.t3.medium) | ~$50 | ~$0.002 |
| API Gateway + CloudFront | ~$30 | ~$0.001 |
| CloudWatch + X-Ray | ~$60 | ~$0.003 |
| **Total (on-demand)** | **~$11,600/month** | **~$0.52/run** |

**With Bedrock Batch Inference (non-urgent jobs — 50% model cost reduction)**

| Component | Monthly (22,500 runs, mixed on-demand/batch) | Per run |
|---|---|---|
| Bedrock models (50% on batch inference) | ~$5,600 | ~$0.25 |
| Infrastructure (same as above) | ~$1,270 | ~$0.056 |
| AgentCore Runtime | ~$675 | ~$0.030 |
| **Total (with batch)** | **~$7,545/month** | **~$0.34/run** |

> Batch inference mode adds ~30–90 minute processing latency in exchange for the 50% model cost reduction. Best for non-real-time use cases (overnight batch processing of drawing sets).

### Cost optimisation levers

| Lever | Savings | Trade-off |
|---|---|---|
| Use claude-sonnet-4-6 for vision instead of opus | ~60% off model cost | Some detection quality reduction on dense drawings |
| Bedrock Batch Inference for async jobs | ~50% off model cost | 30–90 min added latency |
| Cache vision results for duplicate page hashes (Redis) | Eliminates cost for re-runs | Only helps when same pages are submitted multiple times |
| Provisioned Throughput (if volume is predictable) | 10–20% off vs on-demand | Requires 1-month commitment |
| Reduce DPI from 300 → 200 for simple drawings | Smaller images → fewer tokens | May miss fine dimension labels |
| Page range filtering (`--pages 1`) | Only pay for pages actually needed | Requires caller to know which pages have duct plans |

---

## Scenario Comparison

| | Local | Small Scale (AWS) | Mass Scale (AgentCore) |
|---|---|---|---|
| **Volume** | 1–5 runs/day | 15–20 runs/day | 500–1,000 runs/day |
| **Concurrency** | 1 | 3–5 | 50–80 |
| **Cost per run** | ~$0.46 – $0.90 | ~$0.63 | ~$0.34 – $0.52 |
| **Monthly cost** | ~$45–90 | ~$315 | ~$7,500 – $11,600 |
| **Setup complexity** | Minimal | Low–Medium | High |
| **Time to first run** | Minutes | 2–3 days | 2–4 weeks |
| **SLA / availability** | No guarantee | 99.9% (Fargate) | 99.95%+ (AgentCore) |
| **Observability** | CLI logs + CSV | CloudWatch | X-Ray traces + dashboards |
| **Retry handling** | In-process Python | SQS DLQ | AgentCore + Step Functions |
| **Access control** | Local only | API Gateway + API keys | API Gateway + IAM + WAF |
| **Best for** | Development, testing | Internal tools, pilot | Production SaaS, enterprise |

---

## Migration Path

```
Phase 1 (now)
  Local CLI
  ↓
Phase 2 (when volume reaches 5–10/day)
  Containerise the pipeline (Dockerfile)
  Deploy to ECS Fargate
  Add SQS + API Gateway
  ↓
Phase 3 (when volume exceeds ~100/day or SLA requirements tighten)
  Migrate model calls to Amazon Bedrock
  Wrap each agent as a Bedrock AgentCore agent
  Add Step Functions workflow
  Enable batch inference for non-urgent jobs
  Add Redis cache, X-Ray tracing, cost dashboards
```
