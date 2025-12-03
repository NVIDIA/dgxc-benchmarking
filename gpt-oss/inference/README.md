# GPT-OSS 120B Model Inference on Kubernetes

Deploy and benchmark the GPT-OSS 120B model using NVIDIA's Dynamo framework with TensorRT-LLM on Kubernetes clusters with GB200 GPUs. This guide is designed for both Kubernetes beginners and experts to launch and benchmark this architecture.

## 📋 Version Information

| Component | Version | Details |
|-----------|---------|---------|
| **NVIDIA Dynamo** | `0.5.1` | Platform orchestration framework |
| **TensorRT-LLM** | `1.1.0rc2.post2` (uses image :`nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.5.1-rc0.pre3`) | High-performance inference engine |
| **AI Perf** | `pip install aiperf==0.1.1` | NVIDIA benchmarking tool for LLM performance testing |
| **Model** | [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) | GPT-OSS 120B parameter model (117B params, 5.1B active) |
| **Target Hardware** | `4x+ NVIDIA GB200` | Minimum 4 GPUs, scales horizontally (189GB VRAM per GPU) |

**💡 Note:** You can deploy and test the model without running any benchmarking if you choose - just skip the benchmarking steps. See the [Test the Deployment](#-test-the-deployment) section below for manual testing instructions.

<details>
<summary><b>Architecture</b></summary>

```
┌──────────────────────────────────────────── Kubernetes Cluster ──────────────────────────────────────────────┐
│                                                                                                              │
│ ┌──────────────────────────────────────── Namespace: gpt-oss-120b ─────────────────────────────────────────┐ │
│ │                                                                                                          │ │
│ │  ┌─────────────┐    ┌────────────────┐    ┌─────────────────────────────────────┐    ┌────────────────┐  │ │
│ │  │ Benchmark   │───→│    Service     │───→│        DynamoGraphDeployment        │    │      PVC       │  │ │
│ │  │    Job      │    │ gpt-oss-agg-   │    │        gpt-oss-agg                  │    │ model-cache-   │  │ │
│ │  │  (aiperf)   │    │ trtllmworker   │    │ ┌─────────────────────────────────┐ │    │ oss-gpt120b    │  │ │
│ │  │             │    │   Port: 8000   │    │ │     TrtllmWorker Pod(s)         │ │    │ Size: 200Gi    │  │ │
│ │  │             │    │                │    │ │                                 │◄┼────┤ Access: RWX    │  │ │
│ │  │             │    │                │    │ │ • Frontend: HTTP API (:8000)    │ │    │                │  │ │
│ │  │             │    │                │    │ │ • Backend: TensorRT-LLM Engine  │ │    └────────────────┘  │ │
│ │  │             │    │                │    │ │ • GPUs: 4x GB200 GPUs           │ │              │         │ │
│ │  │             │    │                │    │ │ • Memory: 80Gi SharedMem        │ │              │         │ │
│ │  │             │    │                │    │ │ • Mount: /model-cache/          │ │              │         │ │
│ │  └─────────────┘    └────────────────┘    │ └─────────────────────────────────┘ │              │         │ │
│ └────────────────────────────────────────────────────────────────────────────────────────────────┼─────────┘ │
│                                                                                                  │           │
│ ┌─────────────────────────────────── Persistent Storage Backend ───────────────────────────────┐ │           │
│ │                        (NFS/EFS/FSS/Filestore/CephFS)                                        │◄┘           │
│ │                                                                                              │             │
│ │     Model Files: /model-store/hub/models--openai--gpt-oss-120b/                              │             │
│ │     • Config files, tokenizer, model weights (~240GB)                                        │             │
│ │     • TensorRT engines (cached after first build)                                            │             │
│ └──────────────────────────────────────────────────────────────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

External Access (Testing):
                                                    
                                    kubectl port-forward :8000 
                                   (tunnels to Service above)
                                                    ▲
┌──────────────┐                                    │
│   Client(s)  │ ───────────────────────────────────┘
│ (localhost)  │  curl -X POST 'http://localhost:8000/v1/completions' \
│              │    -H 'Content-Type: application/json' \
│              │    -d '{"model":"openai/gpt-oss-120b","prompt":"Hello"}'       
└──────────────┘
```

**Components:**
- **Frontend**: HTTP API server (OpenAI-compatible)
- **Backend**: TensorRT-LLM inference engine
- **Storage**: Shared file system for model caching
- **Secrets**: NGC API key for container registry access

</details>


## 📋 Requirements

### Local Tools
- **kubectl CLI**: Kubernetes command-line tool ([installation guide](https://kubernetes.io/docs/tasks/tools/))
- **helm CLI**: Version 3.0+ for Dynamo platform installation ([installation guide](https://helm.sh/docs/intro/install/))
- **NGC API Key**: Get from [NGC Catalog](https://catalog.ngc.nvidia.com/) → User Account → Setup

### Kubernetes Cluster Requirements
- **GPUs**: 4+ NVIDIA GB200 or equivalent (160GB+ VRAM total)
- **Storage**: Persistent storage for model caching (see [Storage Configuration](#storage-configuration) below)
- **Network**: Internet access for model download from HuggingFace
- **NVIDIA Dynamo**: Framework for LLM deployment (installation covered below)

### Storage Configuration

**⚠️ IMPORTANT: Check with your cluster administrator for the correct storage class and access mode.**

| Storage Type | When to Use | Benefits | Limitations | Example Classes |
|---|---|---|---|---|
| **ReadWriteMany (RWX)** | **Recommended** - Multiple replicas | • Shared model cache<br>• Horizontal scaling<br>• Best performance | • Requires file storage<br>• May cost more | `efs-sc`, `filestore-csi`, `oci-fss` |
| **ReadWriteOnce (RWO)** | Single replica only | • Works with block storage<br>• Simpler setup<br>• Lower cost | • Single replica limit<br>• No cache sharing | `gp2`, `pd-standard`, `oci-bv` |

**Find your storage class:**
```bash
# List available storage classes
kubectl get storageclass

# Check access modes supported
kubectl describe storageclass <storage-class-name> | grep -i access
```

---

## 🚀 Installation Steps

### Step 1: Configure Environment

```bash
# Set your cluster-specific values (customize these)
export NAMESPACE="gpt-oss-120b"                     # Deployment namespace
export STORAGE_CLASS="your-storage-class"           # Check with cluster admin
export NGC_API_KEY="your-ngc-api-key-here"         # From NGC setup

# Validate configuration
echo "Deploying to namespace: $NAMESPACE"
echo "Using storage class: $STORAGE_CLASS"
echo "NGC key configured: ${NGC_API_KEY:0:10}..."
```

> **💡 Variable Usage Note**: When running commands with `&` (background) or in some complex scenarios, shell variable expansion may fail. If you encounter `"flag needs an argument"` errors, use explicit values instead of `$NAMESPACE` (e.g., `--namespace=gpt-oss-120b`).

### Step 2a: Create Namespace and Secrets

```bash
# Create namespace
kubectl create namespace $NAMESPACE

# Create NGC registry secret for pulling NVIDIA containers
kubectl create secret docker-registry nvcrimagepullsecret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password=$NGC_API_KEY \
  -n $NAMESPACE

# Create HuggingFace token secret (empty for public GPT-OSS model)
kubectl create secret generic hf-token-secret \
  --from-literal=HUGGING_FACE_HUB_TOKEN="" \
  -n $NAMESPACE
```

### Step 2b: Check if Dynamo is Already Installed

**Many clusters already have NVIDIA Dynamo installed. Run these commands to check:**

```bash
# Check if Dynamo CRDs are installed
kubectl get crd | grep -E "(dynamo|grove)"
# Expected output should include:
# dynamocomponentdeployments.nvidia.com
# dynamographdeployments.nvidia.com
# podcliques.grove.io (and other grove CRDs)

# Check if Dynamo platform pods are running
kubectl get pods --all-namespaces | grep dynamo
# Expected: dynamo-operator, etcd, nats pods in Running status

# Test that DynamoGraphDeployment resources work
kubectl get dynamographdeployment --all-namespaces
# Should return successfully (may show existing deployments or "No resources found")
```

**✅ If all checks pass:** Dynamo is already installed! Skip to [Create Model Storage](#-create-model-storage)

**❌ If any checks fail:** You need to install Dynamo - continue to the installation section below.

> **💡 Troubleshooting verification:** If commands fail due to permissions, you may not have the necessary cluster access. Contact your cluster administrator or check if you're connected to the right cluster with `kubectl config current-context`.

---

### Step 2c: 🔧 NVIDIA Dynamo Installation (Only if Not Already Installed)

<details>
<summary><b>📦 Click to expand Dynamo installation instructions (only if verification above failed)</b></summary>

### Pre-Installation Cluster Compatibility Check

**Check your cluster for potential compatibility issues before installing Dynamo:**

```bash
# 1. Check cluster architecture mix (important for GB200 clusters)
echo "=== Checking Node Architecture ==="
kubectl get nodes -o custom-columns=NAME:.metadata.name,ARCH:.metadata.labels.'kubernetes\.io/arch' | head -5

# 2. Check for node taints that may block scheduling  
echo "=== Checking Node Taints ==="
kubectl get nodes -o custom-columns=NAME:.metadata.name,TAINTS:.spec.taints | grep -v '<none>' | head -5

# 3. Verify storage class exists and supports your access mode
echo "=== Verifying Storage Class ==="
kubectl get storageclass $STORAGE_CLASS -o custom-columns=NAME:.metadata.name,PROVISIONER:.provisioner,ACCESS:.allowVolumeExpansion
```

**⚠️ Important Notes for Next Steps:**
- If you see **mixed arm64/amd64 architecture**: You'll need to apply architecture fixes in [Step 2e: Apply Cluster-Specific Fixes (If Needed)](#step-2e-apply-cluster-specific-fixes-if-needed)
- If you see **"dedicated: user-workload" taints**: You'll need to apply toleration fixes in [Step 2e: Apply Cluster-Specific Fixes (If Needed)](#step-2e-apply-cluster-specific-fixes-if-needed)
- If storage class doesn't exist: Contact your cluster administrator before proceeding

> **📝 For Novice Users**: Don't worry if you see warnings above - we'll address any issues in [Step 2e: Apply Cluster-Specific Fixes (If Needed)](#step-2e-apply-cluster-specific-fixes-if-needed) after Dynamo is installed. Just note what you see and continue to [Step 2d: Install NVIDIA Dynamo Platform](#step-2d-install-nvidia-dynamo-platform).

### Step 2d: Install NVIDIA Dynamo Platform

**Prerequisites** (verify with your cluster admin):
- Kubernetes cluster version 1.24 or later
- Helm 3.0+ installed
- Cluster admin privileges for CRD installation

**Production Installation**
```bash
# Set Dynamo version - see version info below
export RELEASE_VERSION="0.5.1"  # Current stable version (0.3.2+ supported)

# Install Dynamo CRDs (cluster-wide, requires admin privileges)
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds-${RELEASE_VERSION}.tgz
helm install dynamo-crds dynamo-crds-${RELEASE_VERSION}.tgz --namespace default

# Install Dynamo platform
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz \
  --namespace $NAMESPACE \
  --create-namespace \
  --set "imagePullSecrets[0].name=nvcrimagepullsecret" \
  --set "grove.enabled=true"
```

> **💡 Important**: The `grove.enabled=true` flag enables Grove, a required component for model deployment orchestration. By default, Grove and Kai Scheduler are disabled in Dynamo installations, but they are necessary for proper functioning of DynamoGraphDeployments.

> **📖 For comprehensive installation options**, including local development (Minikube), custom builds, troubleshooting, and advanced configurations, see the official [NVIDIA Dynamo Installation Guide](https://docs.nvidia.com/dynamo/latest/index.html) and the [Step 2c: NVIDIA Dynamo Installation](#step-2c--nvidia-dynamo-installation-only-if-not-already-installed) section.

**Verify Dynamo Installation:**
```bash
# Check that CRDs are installed
kubectl get crd | grep dynamo

# Check that platform pods are running  
kubectl get pods -n $NAMESPACE
# Expected: dynamo-operator-*, etcd-*, nats-* pods in Running status

# Test that DynamoGraphDeployment resources can be created
kubectl get dynamographdeployment -n $NAMESPACE
# Should return "No resources found" (not an error)
```

**⚠️ CRITICAL: Verify Grove CRD Compatibility**

The Dynamo operator requires specific Grove CRDs. Check for compatibility issues:

```bash
# Check if Dynamo operator is healthy
kubectl get pods -n $NAMESPACE | grep dynamo-operator
# Should show "Running" status, NOT "CrashLoopBackOff"

# If operator is crashing, check for Grove CRD compatibility
kubectl logs deployment/dynamo-platform-dynamo-operator-controller-manager -n $NAMESPACE --tail=5
# Look for error: "no matches for kind \"PodGangSet\" in version \"grove.io/v1alpha1\""
```

**🔧 Grove CRD Compatibility Fix (If Needed)**

If you see the `PodGangSet` error above, apply this fix:

```bash
# Check what Grove CRDs are available
kubectl get crd | grep grove

# If you see "podgangs.scheduler.grove.io" but NOT "podgangsets.grove.io", apply this fix:
kubectl get crd podgangs.scheduler.grove.io -o yaml | \
  sed 's/scheduler\.grove\.io/grove.io/g; s/PodGang/PodGangSet/g; s/podgangs/podgangsets/g' | \
  kubectl apply -f -

# Restart the Dynamo operator to pick up the new CRD
kubectl rollout restart deployment/dynamo-platform-dynamo-operator-controller-manager -n $NAMESPACE

# Wait for operator to be healthy
kubectl rollout status deployment/dynamo-platform-dynamo-operator-controller-manager -n $NAMESPACE --timeout=120s

# Verify fix worked
kubectl get pods -n $NAMESPACE | grep dynamo-operator
# Should now show "Running" status
```

### Step 2e: Apply Cluster-Specific Fixes (If Needed)

**Only apply the fixes below if you identified issues in [Step 2c: NVIDIA Dynamo Installation](#step-2c--nvidia-dynamo-installation-only-if-not-already-installed). If you didn't see any compatibility warnings, skip to [Create Model Storage](#-create-model-storage).**

**🔧 Fix A: Mixed Architecture Clusters (if you saw arm64 + amd64 in [Step 2c: NVIDIA Dynamo Installation](#step-2c--nvidia-dynamo-installation-only-if-not-already-installed))**

GB200 clusters often have ARM64 GPU nodes and x86_64 CPU nodes. The Dynamo operator must run on x86_64:

```bash
# Force Dynamo operator to run on x86_64 nodes only
echo "Applying architecture fix..."
kubectl patch deployment dynamo-platform-dynamo-operator-controller-manager -n $NAMESPACE \
  --type='merge' -p='{"spec":{"template":{"spec":{"affinity":{"nodeAffinity":{"requiredDuringSchedulingIgnoredDuringExecution":{"nodeSelectorTerms":[{"matchExpressions":[{"key":"kubernetes.io/arch","operator":"In","values":["amd64"]}]}]}}}}}}}'

# Verify the fix was applied
echo "Verifying architecture fix..."
kubectl get deployment dynamo-platform-dynamo-operator-controller-manager -n $NAMESPACE -o jsonpath='{.spec.template.spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].values[0]}'
echo ""
```

**🔧 Fix B: Tainted Nodes (if you saw "dedicated: user-workload" taints in [Step 2c: NVIDIA Dynamo Installation](#step-2c--nvidia-dynamo-installation-only-if-not-already-installed))**

Some clusters taint GPU nodes to reserve them for user workloads. Add tolerations to all Dynamo components:

```bash
# Add tolerations to Dynamo deployments
echo "Applying taint tolerations to deployments..."
for component in dynamo-platform-dynamo-operator-controller-manager dynamo-platform-nats-box; do
  echo "  - Patching deployment: $component"
  kubectl patch deployment $component -n $NAMESPACE \
    --type='merge' -p='{"spec":{"template":{"spec":{"tolerations":[{"effect":"NoSchedule","key":"dedicated","operator":"Equal","value":"user-workload"},{"effect":"NoExecute","key":"dedicated","operator":"Equal","value":"user-workload"}]}}}}'
done

# Add tolerations to Dynamo statefulsets  
echo "Applying taint tolerations to statefulsets..."
for component in dynamo-platform-etcd dynamo-platform-nats; do
  echo "  - Patching statefulset: $component"
  kubectl patch statefulset $component -n $NAMESPACE \
    --type='merge' -p='{"spec":{"template":{"spec":{"tolerations":[{"effect":"NoSchedule","key":"dedicated","operator":"Equal","value":"user-workload"},{"effect":"NoExecute","key":"dedicated","operator":"Equal","value":"user-workload"}]}}}}'
done

# Verify the fixes were applied
echo "Verifying taint fixes..."
kubectl get pods -n $NAMESPACE -o custom-columns=NAME:.metadata.name,STATUS:.status.phase,NODE:.spec.nodeName
```

**✅ Verification: Ensure All Dynamo Pods Are Running**

After applying any fixes, verify all Dynamo components are healthy:

```bash
# Wait for all pods to be ready (may take 2-3 minutes after applying fixes)
echo "Waiting for Dynamo components to be ready..."
kubectl wait --for=condition=ready pod --all -n $NAMESPACE --timeout=300s

# Final status check
echo "Final Dynamo platform status:"
kubectl get pods -n $NAMESPACE
kubectl get dynamographdeployment -n $NAMESPACE
```

> **💡 Troubleshooting**: If pods remain in `Pending` status after 5 minutes, run `kubectl describe pod <pod-name> -n $NAMESPACE` to see specific scheduling issues.

**✅ End of Dynamo Installation Section**

</details>

### Step 3: Install [yq](https://github.com/mikefarah/yq)

[yq](https://github.com/mikefarah/yq) is a command-line editor for processing YAMLs.
Please follow the installation instructions [here](https://github.com/mikefarah/yq?tab=readme-ov-file#install) for your specific platform and architecture.
```bash
# Test the installation of yq
yq --version
# Example output:
# yq (https://github.com/mikefarah/yq/) version v4.48.1
```

---

### 📁 Create Model Storage

```bash
# Create PVC for GPT-OSS model cache (240GB+ model)
kubectl apply -n $NAMESPACE -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
  namespace: $NAMESPACE
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 200Gi
  storageClassName: $STORAGE_CLASS
EOF

# Verify PVC is bound
kubectl get pvc -n $NAMESPACE
```

> **💡 Using ReadWriteOnce (RWO) storage?** See [RWO Configuration](#rwo-storage-configuration) for required changes.

### ⬇️ Download the Model

```bash
kubectl apply -n $NAMESPACE -f model_cache/model-download.yaml

# Check for status
kubectl get -n $NAMESPACE jobs | grep model-download
kubectl get -n $NAMESPACE pods | grep model-download

# Follow the model download logs
kubectl logs -n $NAMESPACE -f $(kubectl get pods -n $NAMESPACE | grep 'model-download-.*' | awk '{print $1}')
```

Model download could take 10-15 mins.

### 🚀 Deploy the GPT-OSS Model

```bash
# Deploy configuration, model, and services
kubectl apply -n $NAMESPACE -f config.yaml
kubectl apply -n $NAMESPACE -f deploy.yaml
yq e '.metadata.namespace = $NAMESPACE' -i service.yaml
kubectl apply -f service.yaml
yq e '.metadata.namespace = $NAMESPACE' -i frontend-service.yaml
kubectl apply -f frontend-service.yaml
```

### 👀 Monitor Deployment

```bash
# Check deployment status
kubectl get dynamographdeployment -n $NAMESPACE
kubectl get pods -n $NAMESPACE

# Monitor model loading (takes 15-30 minutes for first deployment)
kubectl logs -n $NAMESPACE -f $(kubectl get pods -n $NAMESPACE | grep 'gpt-oss-agg.*trtllmworker' | awk '{print $1}')

# Wait for deployment to be ready
# Typically this means waiting for the frontend pod to start up
kubectl wait --for=condition=ready pod $(kubectl get pods -n $NAMESPACE | grep 'gpt-oss-agg.*frontend' | awk '{print $1}') -n $NAMESPACE --timeout=1800s
```

#### What You Should See During Deployment

**Phase 1: DynamoGraphDeployment Creation (30 seconds)**
```bash
kubectl get dynamographdeployment -n $NAMESPACE
# Example output:
# NAME              READY   STATUS   AGE
# gpt-oss-agg       False   Pending  15s
```

**Phase 2: Pod Creation and Initialization (2-5 minutes)**
```bash
kubectl get pods -n $NAMESPACE
# Example output:
# NAME                                READY   STATUS            RESTARTS   AGE
# gpt-oss-agg-trtllmworker-0     0/1     ContainerCreating 0          1m
```

**Phase 3: Loading (10-15 minutes)**

Pod status changes to `Running` but `READY` remains `0/1`:

```bash
kubectl get pods -n $NAMESPACE
# Example output:
# NAME                               READY   STATUS    RESTARTS   AGE
# gpt-oss-agg-0-frontend-25zhj       1/1     Running     0        20h
# gpt-oss-agg-0-trtllmworker-lhv25   1/1     Running     0        20h
```

**Key log messages to look for:**
Logs in trtllm worker
```bash
# Initial startup
[10/21/2025-23:23:13] [TRT-LLM] [I] TensorRT-LLM inited.
[TensorRT-LLM] TensorRT-LLM version: 1.1.0rc2.post2

# TensorRT engine
[10/21/2025-23:23:13] [TRT-LLM] [I] Starting TensorRT-LLM init.
[TensorRT-LLM][INFO] Set logger level to INFO
[10/21/2025-23:23:13] [TRT-LLM] [I] TensorRT-LLM inited.

# Engine loading and model registration (2-3 minutes)
[10/21/2025-23:23:27] [TRT-LLM] [RANK 0] [I] Prefetching /model-store/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/model-00004-of-00014.safetensors to memory...

# Run CUDA Graph warmup
[10/21/2025-23:24:47] [TRT-LLM] [RANK 0] [I] Run generation only CUDA graph warmup for batch size=512, draft_len=0
[10/21/2025-23:24:47] [TRT-LLM] [RANK 3] [I] Run generation only CUDA graph warmup for batch size=512, draft_len=0

# Done initialization
[10/21/2025-23:25:18] [TRT-LLM] [RANK 0] [I] Setting PyTorch memory fraction to 0.21594039253566577 (39.7330322265625 GiB)
[10/21/2025-23:25:18] [TRT-LLM] [RANK 1] [I] Setting PyTorch memory fraction to 0.21994681980298914 (40.47021484375 GiB)
[10/21/2025-23:25:18] [TRT-LLM] [RANK 2] [I] Setting PyTorch memory fraction to 0.2199398538340693 (40.46893310546875 GiB)
[10/21/2025-23:25:18] [TRT-LLM] [RANK 3] [I] Setting PyTorch memory fraction to 0.22078074579653534 (40.6236572265625 GiB)
```

Logs for the frontend worker should show
```
# Frontend startup
2025-10-21T23:22:52.381069Z  INFO dynamo_llm::entrypoint::input::http: Watching for remote model at models
2025-10-21T23:22:52.382010Z  INFO dynamo_llm::http::service::service_v2: Starting HTTP(S) service protocol="HTTP" address="0.0.0.0:8000"
2025-10-21T23:22:52.382102Z  INFO dynamo_llm::http::service::service_v2: chat endpoints enabled
2025-10-21T23:22:52.382114Z  INFO dynamo_llm::http::service::service_v2: completion endpoints enabled
2025-10-21T23:22:54.322550Z  INFO dynamo_llm::discovery::watcher: added model model_name="openai/gpt-oss-120b"
2025-10-21T23:25:18.499322Z  INFO dynamo_llm::http::service::service_v2: chat endpoints enabled
2025-10-21T23:25:18.499355Z  INFO dynamo_llm::http::service::service_v2: completion endpoints enabled
2025-10-22T19:03:14.785269Z  INFO dynamo_llm::http::service::service_v2: chat endpoints disabled
2025-10-22T19:03:14.785307Z  INFO dynamo_llm::http::service::service_v2: completion endpoints disabled
2025-10-22T19:06:59.043561Z  INFO dynamo_llm::http::service::service_v2: chat endpoints enabled
2025-10-22T19:06:59.043589Z  INFO dynamo_llm::http::service::service_v2: completion endpoints enabled
```

**Phase 4: Ready State**

DynamoGraphDeployment shows `READY=True`:
```bash
kubectl get dynamographdeployment -n $NAMESPACE
# Example output:
# NAME          READY   STATUS   AGE
# gpt-oss-agg   True    Running  25m
```

Pod shows `READY=1/1`:
```bash
kubectl get pods -n $NAMESPACE
# Example output:
# NAME                           READY   STATUS    RESTARTS   AGE
# gpt-oss-agg-trtllmworker-0     1/1     Running   0          25m
```

Service has endpoints:
```bash
kubectl get endpoints -n $NAMESPACE
# Example output:
# NAME                      ENDPOINTS         AGE
# gpt-oss-agg-trtllmworker   10.244.0.15:8000  25m
```

<details>
<summary><b>🔧 Troubleshooting Common Issues</b></summary>

**Pod stuck in `Pending`:**
```bash
kubectl describe pod <pod-name> -n $NAMESPACE
# Common causes:
# - Insufficient GPUs: "0/14 nodes are available: 9 Insufficient nvidia.com/gpu"
# - Taint issues: "0/14 nodes are available: 11 node(s) had untolerated taint {dedicated: user-workload}"
# - PVC binding: "pod has unbound immediate PersistentVolumeClaims"
```

**Pod in `CrashLoopBackOff`:**
```bash
kubectl logs <pod-name> -n $NAMESPACE
# Common causes:
# - Config file not found: "FileNotFoundError: .../config.yaml"
# - HF authentication: "401 Client Error: Unauthorized"
# - Memory issues: "CUDA out of memory"
```

**Model loading stuck:**
```bash
# If logs stop updating for >10 minutes, check:
kubectl describe pod <pod-name> -n $NAMESPACE | grep Events
kubectl top pod <pod-name> -n $NAMESPACE  # Check resource usage
```

</details>

<details>
<summary><b>📋 Expected Timeline</b></summary>

| Phase | Duration | What's Happening |
|-------|----------|------------------|
| **Pod Creation** | 30s-2min | K8s scheduling, image pull |
| **Model Download** | 5-10min | HuggingFace download (first time only) |
| **Engine Build** | 10-15min | TensorRT optimization |
| **Model Loading** | 2-3min | GPU memory allocation |
| **Total (first time)** | **20-30min** | **Full cold start** |
| **Total (cached)** | **5-10min** | **Subsequent deployments** |

</details>

---

## 🧪 Test the Deployment

```bash
# Set up port forwarding to access the API
# Note: Use explicit namespace to avoid variable expansion issues in background processes
kubectl port-forward service/gpt-oss-agg-frontend 8000:8000 --namespace=gpt-oss-120b &

# Wait a moment for port forwarding to establish
sleep 3

# Test 1: Check available models
curl -X GET 'http://localhost:8000/v1/models'
# Expected: {"object":"list","data":[{"id":"openai/gpt-oss-120b"...}]}

# Test 2: Generate text
curl -X POST 'http://localhost:8000/v1/completions' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openai/gpt-oss-120b",
    "prompt": "The future of AI is",
    "max_tokens": 50
  }'
# Expected: {"id":"cmpl-...","choices":[{"text":"..."}]...}

# To stop port forwarding when done:
pkill -f "kubectl port-forward"
```

<details>
<summary><b>🔧 Port Forwarding Troubleshooting</b></summary>

**Issue**: `error: flag needs an argument: 'n' in -n`
```bash
# ❌ This fails with background processes:
kubectl port-forward service/gpt-oss-agg-frontend 8000:8000 -n $NAMESPACE &

# ✅ Use explicit namespace instead:
kubectl port-forward service/gpt-oss-agg-frontend 8000:8000 --namespace=gpt-oss-120b &
```

**Issue**: `curl: (7) Failed to connect to localhost port 8000`
```bash
# Check if port forwarding is active:
ps aux | grep "kubectl port-forward"

# If not running, restart it:
kubectl port-forward service/gpt-oss-agg-frontend 8000:8000 --namespace=gpt-oss-120b &
sleep 3
```

**Issue**: `namespace "..." not found`
```bash
# Verify your namespace exists and is correctly named:
kubectl get namespaces | grep gpt-oss
# Should show: gpt-oss-120b

# Double-check your deployment:
kubectl get pods --namespace=gpt-oss-120b
```

</details>

---

## 📊 Run Benchmark (25-30 minutes)

> **⏰ Timing**: Benchmark includes extensive warmup for accurate performance measurement. Allow 25-30 minutes for completion.

```bash
# Run performance benchmark  
kubectl apply -f bench.yaml -n $NAMESPACE

# Monitor progress - you'll see three phases:
# 1. Warmup: ~20 minutes (13,500+ requests)
# 2. Profiling: ~5 minutes (performance measurement) 
# 3. Results: ~1 minute (report generation)
kubectl logs -f job/oss-gpt120b-bench -n $NAMESPACE
```

### 📊 Getting Benchmark Results

**Check benchmark completion:**
```bash
# Check if benchmark is complete
kubectl get jobs -n $NAMESPACE
# Look for: STATUS=Complete, COMPLETIONS=1/1
```

**View results in terminal:**
```bash
# Benchmark pod may finish, use the trtllm worker pod to access results from the volume
POD_NAME=$(kubectl get pods -n $NAMESPACE | grep 'gpt-oss-agg.*trtllmworker' | awk '{print $1}')
PERF_DIR=$(kubectl exec $POD_NAME -n $NAMESPACE -- ls -t /model-store/perf/ | head -1)
CONCURRENCY_DIR=$(kubectl exec $POD_NAME -n $NAMESPACE -- ls -t /model-store/perf/$PERF_DIR/ | head -1)

# View CSV results in terminal
kubectl exec $POD_NAME -n $NAMESPACE -- cat /model-store/perf/$PERF_DIR/$CONCURRENCY_DIR/profile_export_aiperf.csv

# View JSON results in terminal  
kubectl exec $POD_NAME -n $NAMESPACE -- cat /model-store/perf/$PERF_DIR/$CONCURRENCY_DIR/profile_export_aiperf.json
```

**Download result files (optional):**
```bash
# Copy CSV results file to local machine
kubectl cp $POD_NAME:/model-store/perf/$PERF_DIR/$CONCURRENCY_DIR/profile_export_aiperf.csv ./benchmark_results.csv -n $NAMESPACE

# Copy JSON results file to local machine
kubectl cp $POD_NAME:/model-store/perf/$PERF_DIR/$CONCURRENCY_DIR/profile_export_aiperf.json ./benchmark_results.json -n $NAMESPACE
```

---

<details>
<summary><b>⚙️ Additional Configurations for Specific Clusters</b></summary>

### RWO Storage Configuration

If you only have ReadWriteOnce storage:

```bash
# Modify PVC for RWO (single replica only)
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
  namespace: $NAMESPACE
spec:
  accessModes:
  - ReadWriteOnce  # Changed from ReadWriteMany
  resources:
    requests:
      storage: 200Gi
  storageClassName: $STORAGE_CLASS
EOF

# Ensure deployment uses single replica (already default in deploy.yaml)
# No changes needed to deploy.yaml as replicas: 1 is already set
```

### Namespace Restrictions

If Dynamo operator is restricted to specific namespaces:

```bash
# Check for restrictions
kubectl get deployment dynamo-platform-dynamo-operator-controller-manager -n $NAMESPACE -o yaml | grep restrictedNamespace

# Remove restriction if present (requires admin)
kubectl patch deployment dynamo-platform-dynamo-operator-controller-manager -n $NAMESPACE \
  --type='json' -p='[{"op": "remove", "path": "/spec/template/spec/containers/0/args"}]'
```

</details>

---

<details>
<summary><b>🚨 Troubleshooting</b></summary>

### Critical Issues

**🔥 Dynamo Operator CrashLoopBackOff (Grove CRD Issue)**

**Symptoms:**
```bash
kubectl get pods -n $NAMESPACE | grep dynamo-operator
# Shows: CrashLoopBackOff instead of Running
```

**Diagnosis:**
```bash
kubectl logs deployment/dynamo-platform-dynamo-operator-controller-manager -n $NAMESPACE --tail=10
# Look for: "no matches for kind \"PodGangSet\" in version \"grove.io/v1alpha1\""
```

**Fix:**
This is a Grove CRD compatibility issue. Apply the fix from Step 3:
```bash
kubectl get crd podgangs.scheduler.grove.io -o yaml | \
  sed 's/scheduler\.grove\.io/grove.io/g; s/PodGang/PodGangSet/g; s/podgangs/podgangsets/g' | \
  kubectl apply -f -
kubectl rollout restart deployment/dynamo-platform-dynamo-operator-controller-manager -n $NAMESPACE
```

### Common Issues

**Pod stuck in `Pending`:**
```bash
kubectl describe pod <pod-name> -n $NAMESPACE
# Check for: insufficient GPUs, PVC binding issues, node taints
```

**Service has no endpoints:**
```bash
kubectl get endpoints -n $NAMESPACE
# If empty, check service selector matches pod labels
```

**Model loading fails:**
```bash
kubectl logs <pod-name> -n $NAMESPACE
# Check for: authentication errors, storage issues, memory problems
```

**Benchmark architecture errors:**
```bash
# Ensure benchmark runs on x86_64 CPU nodes (already configured in bench.yaml)
kubectl describe pod <benchmark-pod> -n $NAMESPACE | grep "Node:"
```

### Diagnostic Commands

```bash
# Full cluster status
kubectl get all -n $NAMESPACE
kubectl get dynamographdeployment -n $NAMESPACE
kubectl get pvc -n $NAMESPACE

# Resource usage
kubectl top pods -n $NAMESPACE
kubectl top nodes
```

</details>

---

## 🗑️ Cleanup

```bash
# Delete GPT-OSS deployment
kubectl delete -f config.yaml -f deploy.yaml -f service.yaml -f frontend-service.yaml

# Delete benchmark job
kubectl delete -f bench.yaml

# Delete PVC (⚠️ removes cached model files)
kubectl delete pvc model-cache -n $NAMESPACE

# Delete namespace (removes everything)
kubectl delete namespace $NAMESPACE
```

---

## 🚀 Scaling Guide

Scale your GPT-OSS deployment to any size from 1-18 nodes (4-72 GPUs). Simply set your desired scale and copy/paste the commands below.

### 🎯 Set Your Scale (Change This Value)

```bash
# ⚡ SET YOUR DESIRED SCALE HERE ⚡
# Supported range: 1-18 nodes (4-72 GPUs total)
export NUM_NODES=1

# 📊 Calculated values (don't change these)
export TOTAL_GPUS=$((NUM_NODES * 4))
```

### 🚀 Deploy at Scale

```bash
# Scale deployment to desired size
kubectl patch dynamographdeployment gpt-oss-agg -n $NAMESPACE --type='merge' -p='{
  "spec": {
    "services": {
      "Frontend": {"replicas": '$NUM_NODES'},
      "TrtllmWorker": {"replicas": '$NUM_NODES'}
    }
  }
}'

echo "🔄 Scaling to $NUM_NODES nodes ($TOTAL_GPUS GPUs total)..."
echo "⏱️  This may take 5-15 minutes depending on scale..."

# Wait for all pods to be ready
kubectl wait --for=condition=ready pod -l nvidia.com/dynamo-graph-deployment-name=gpt-oss-agg -n $NAMESPACE --timeout=1200s

# Verify deployment
echo "✅ Deployment Status:"
kubectl get dynamographdeployment -n $NAMESPACE
kubectl get pods -n $NAMESPACE | grep -E "(frontend|trtllmworker)" | wc -l | xargs echo "Total pods:"
```

### 🧪 Benchmark at Scale

```bash
# Create optimized benchmark job for your scale
kubectl get job oss-gpt120b-bench -n $NAMESPACE -o yaml | \
yq -e '
  del(.metadata.uid, .metadata.resourceVersion, .metadata.creationTimestamp, .metadata.managedFields, .status) |
  del(.metadata.labels."controller-uid", .metadata.labels."batch.kubernetes.io/controller-uid", .metadata.labels."batch.kubernetes.io/job-name") |
  del(.spec.selector) |
  del(.spec.template.metadata.creationTimestamp) |
  del(.spec.template.metadata.labels."controller-uid", .spec.template.metadata.labels."batch.kubernetes.io/controller-uid", .spec.template.metadata.labels."job-name", .spec.template.metadata.labels."batch.kubernetes.io/job-name") |
  .metadata.name = "oss-gpt120b-bench-" + env(NUM_NODES) + "node" |
  .spec.template.metadata.labels.app = "oss-gpt120b-bench-" + env(NUM_NODES) + "node" |
  (.spec.template.spec.containers[].env[] | select(.name == "DEPLOYMENT_GPU_COUNT")).value = env(TOTAL_GPUS) + ""
' | \
kubectl apply -f -

echo "🏃 Benchmark started for $NUM_NODES nodes ($TOTAL_GPUS GPUs)"
echo "📊 Monitor progress:"
echo "   kubectl logs -f job/oss-gpt120b-bench-${NUM_NODES}node -n $NAMESPACE"

# Quick status check
kubectl get jobs -n $NAMESPACE | grep bench
```

### 🔄 Scale Back Down

```bash
# Return to single-node configuration
export NUM_NODES=1
kubectl patch dynamographdeployment gpt-oss-agg -n $NAMESPACE --type='merge' -p='{
  "spec": {
    "services": {
      "Frontend": {"replicas": 1},
      "TrtllmWorker": {"replicas": 1}
    }
  }
}'

# Clean up benchmark jobs
kubectl delete jobs -l app=oss-gpt120b-bench -n $NAMESPACE

echo "📉 Scaled back to single node"
```

### 💡 Performance Tips

- **🎯 Run to Run Variability**: You can tweak the "-request-rate" flag of AIPerf in bench.yaml. A rate of R over concurrency C will mean the load generate takes C/R seconds to ramp up load. We found tuning this at higher concurrencies is helpful.
- **📈 Linear scaling**: Performance scales linearly with GPU count
- **⏱️ Warmup time**: Increases with scale (allow extra time for large deployments)
- **⚡ Stable concurrency**: 1k per node avoids connection timeouts
- **🔧 Connection limits**: Auto-adjusted for scales >8 nodes
- **📊 Monitoring**: Use `kubectl top nodes` to monitor cluster resource utilization

---



## 📚 Additional Resources

- **Model Information**: [GPT-OSS on HuggingFace](https://huggingface.co/openai/gpt-oss-120b)
- **NVIDIA Dynamo**: [Documentation](https://docs.nvidia.com/nim/)
- **TensorRT-LLM**: [GitHub Repository](https://github.com/NVIDIA/TensorRT-LLM)
- **Kubernetes**: [Official Documentation](https://kubernetes.io/docs/home/)



