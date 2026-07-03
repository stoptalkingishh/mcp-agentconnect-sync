# Adapter Inventory

This repository has three distinct adapter surfaces:

1. Cloud provider call adapters in the router gateway.
2. Local inference backends in the model manager.
3. Rental / node lifecycle adapters for rented GPU provisioning.

The list below separates those surfaces so “adapter” means something concrete.

## 1. Cloud Provider Call Adapters

These determine how the router turns a generic generation request into a provider-specific API call.

| Adapter | Status | Where | Purpose | Notes |
|---|---|---|---|---|
| OpenAI-compatible cloud call | Implemented | `packages/agentconnect-router/src/agentconnect/router/gateway.py` | Calls `/chat/completions` with OpenAI-shaped payloads | Works for OpenAI, Groq, and other compatible endpoints |
| Gemini native adapter | Missing | N/A | Translate router requests to Gemini’s native API | Needed because Gemini is not OpenAI-shaped |
| Anthropic native adapter | Missing | N/A | Translate router requests to Anthropic’s Messages API | Same reason: request/response shape differs |
| OpenRouter adapter | Missing | N/A | Call OpenRouter’s API directly or through compatible routing | Useful if you want provider multiplexing behind one endpoint |
| Bedrock adapter | Missing | N/A | Use AWS Bedrock model invocation APIs | Requires its own auth and payload mapping |
| Azure OpenAI adapter | Missing | N/A | Call Azure-hosted OpenAI endpoints | Endpoint and auth differ from raw OpenAI |
| Mistral adapter | Missing | N/A | Call Mistral’s API directly | Native API shape is not guaranteed to match OpenAI-compatible behavior |
| Cohere adapter | Missing | N/A | Call Cohere’s native generation/chat APIs | Useful if you want native Cohere features or tool semantics |
| xAI adapter | Missing | N/A | Call xAI’s API directly | Separate auth and model naming from OpenAI-compatible providers |
| DeepSeek adapter | Missing | N/A | Call DeepSeek’s API directly | May be OpenAI-compatible in some deployments, but a native adapter keeps the boundary explicit |
| Perplexity adapter | Missing | N/A | Call Perplexity’s API directly | Useful if the provider exposes provider-specific search/chat behavior |
| Vertex AI adapter | Missing | N/A | Call Google Cloud Vertex AI model endpoints | Distinct from Gemini native API, depending on deployment path |
| Hugging Face inference adapter | Missing | N/A | Call HF Inference Endpoints / hosted model APIs | Good for arbitrary hosted models with provider-specific routing |
| Local/stub cloud adapter | Implemented as fallback | `gateway.py` | Returns deterministic stub output when cloud is unavailable | Good for offline demos, not a real provider adapter |

### Cloud Provider Classes

Many “models” are actually accessed through one of these provider classes:

- OpenAI-compatible chat APIs: OpenAI, Groq, Together, Fireworks, many self-hosted gateways
- Native chat/generation APIs: Gemini, Anthropic, Cohere, Mistral, xAI, Perplexity
- Cloud model platforms: AWS Bedrock, Google Vertex AI, Hugging Face endpoints

If a provider speaks the OpenAI-compatible shape, you can usually cover it with the existing adapter. If it does not, it needs its own adapter or a compatibility shim.

### Broker / Router Layers

These are not model providers in the same sense. They are aggregation layers that can usually be covered by one adapter if they expose an OpenAI-compatible surface:

| Layer | Status | Adapter need | Notes |
|---|---|---|---|
| OpenRouter | Implemented by compatibility, not first-class here | Usually no separate adapter | Use the existing OpenAI-compatible cloud adapter unless you need OpenRouter-specific routing metadata |
| LiteLLM | Implemented by compatibility, not first-class here | Usually no separate adapter | LiteLLM explicitly presents a unified OpenAI-style interface and translates to many provider endpoints |
| 9router | Likely broker/proxy | Usually no separate adapter if OpenAI-compatible | Treat as a proxy layer unless its API shape differs |
| OmniRoute | Likely broker/proxy | Usually no separate adapter if OpenAI-compatible | Same rule as other routers/proxies |
| cliproxyapi | Likely proxy | Usually no separate adapter if OpenAI-compatible | Same rule as other routers/proxies |
| Multi AI free model router | Broker/router | Usually no separate adapter | The public page describes it as a free models router hosted by OpenRouter |

### Provider Families From BenchLM / Multi-AI

These names are usually provider or model-family labels, not adapter classes. Whether they need a dedicated adapter depends on whether you call them directly or through a broker like LiteLLM/OpenRouter.

| Family | Adapter need | Practical rule |
|---|---|---|
| OpenAI, Anthropic, xAI, Azure OpenAI, Vertex AI, NVIDIA, Hugging Face, Ollama, OpenRouter | Usually no separate adapter if using LiteLLM or an OpenAI-compatible surface | Reuse the existing OpenAI-compatible adapter or LiteLLM proxy adapter |
| Alibaba / Qwen, Z.AI, Moonshot AI, DeepSeek, MiniMax, Xiaomi, Sarvam, Mistral, Cohere, DeepReinforce, H Company, Meituan, Interfaze, StepFun, LG AI Research, Zyphra, Tencent, Tencent Hunyuan, SK Telecom, Naver Cloud, Kakao, LightOn, Aleph Alpha, IBM, ByteDance, Aion Labs, Upstage, Prism ML, Poolside, LiquidAI, Inception, NC AI, OpenBMB, Cursor, Arcee AI, Community, Academic | Maybe, depending on transport | If the provider exposes a native API, add a native adapter; if it is reachable via LiteLLM/OpenRouter/OpenAI-compatible endpoints, fold it into the existing adapter |
| Runway, Stability AI | Usually yes, but for media generation rather than chat | These are generally not plain chat LLMs, so they deserve separate image/video generation adapters if you want first-class support |

### AI SDK Reference Map

The AI SDK site is useful as a cross-check for provider families and compatibility boundaries. It presents itself as a “unified AI layer” and lists broad provider support, including OpenAI, Anthropic, xAI, Vertex AI, NVIDIA, Hugging Face, Azure OpenAI, Ollama, and OpenRouter. Source: [AI SDK](https://ai-sdk.dev/).

Use it as a reference map, not as a direct implementation guide for this repo:

- If a provider is exposed through the AI SDK as OpenAI-compatible or via a common chat interface, it usually fits the existing adapter.
- If the provider is only supported through SDK-specific translation or has a native API shape, it likely needs its own adapter here.
- If the provider is really a router or gateway, treat it as a broker layer, not a per-model adapter.

In short, the AI SDK reinforces the same split this repo needs:

- provider-native adapters for non-OpenAI APIs
- broker adapters for routing layers
- reuse of the existing OpenAI-compatible path when the surface already matches

### AI SDK Tools Registry

The AI SDK tools registry is also relevant, but for a different reason: it is about **agent tools**, not model transport. The registry lists prebuilt tools for code execution, web search, extraction, browser automation, guardrails, and data connectors. Source: [AI SDK Tools Registry](https://ai-sdk.dev/resources/tools).

That matters here because it suggests a separate adapter layer for:

- tool execution
- browser automation
- web search / extraction
- data-source connectors

Those are not model adapters. They are **tool adapters**. If this repo grows beyond plain model routing, keep those separate from model/provider integration so the architecture stays readable.

### Adapter Decision Rule

- If the service exposes an OpenAI-compatible chat/completions API, reuse the current adapter.
- If the service is a router/proxy, add one adapter for the router layer, not one per upstream model.
- If the service exposes a native non-OpenAI API, add a provider-specific adapter.
- If the service is a media generator, treat it as a separate generation surface, not a chat model.

## 2. Local Inference Backends

These run inside the local model manager and decide how the model server generates text.

| Adapter | Status | Where | Purpose | Notes |
|---|---|---|---|---|
| Stub backend | Implemented | `packages/agentconnect-model-manager/src/agentconnect/model_manager/backends.py` | Deterministic offline generation | Used for tests and demos |
| OpenAI-compatible backend | Implemented | `backends.py` | Calls a local OpenAI-compatible server | Covers vLLM, llama.cpp server, Ollama, SGLang, TGI when exposed compatibly |
| vLLM backend | Covered indirectly | `OpenAICompatibleBackend` | Serve a model through vLLM’s OpenAI-compatible endpoint | No separate adapter needed unless you want vLLM-specific features |
| llama.cpp backend | Covered indirectly | `OpenAICompatibleBackend` | Serve a model through llama.cpp’s OpenAI-compatible server | Same idea: covered if the server speaks compatible chat completions |
| Ollama backend | Covered indirectly | `OpenAICompatibleBackend` | Call Ollama through its OpenAI-compatible API | Native Ollama features are not modeled yet |
| SGLang backend | Covered indirectly | `OpenAICompatibleBackend` | Call SGLang via compatible completions API | Same constraint as the other compatible servers |
| Native model-manager backend interface | Implemented | `backends.py` | Abstract interface for new backends | Implementations can replace the HTTP client behavior entirely |

## 3. Rented-Node Provisioning Adapters

These provision and retire rented GPU instances that host the local model manager.

| Adapter | Status | Where | Purpose | Notes |
|---|---|---|---|---|
| Stub provisioner | Implemented | `packages/agentconnect-router/src/agentconnect/router/provisioning.py` | Deterministic offline rented-node lifecycle | Used for tests and demos |
| RunPod provisioner | Implemented | `provisioning.py` | Rent, poll, and terminate RunPod instances | Exercise path exists and is tested offline |
| Lambda / Vast provisioners | Missing | N/A | Vendor-specific rental control planes | Mentioned in config/docs, not implemented yet |
| Generic provisioner factory | Implemented | `provisioning.py` | Select provisioner by vendor | Falls back to stub for `generic` |

## 4. What “Adapter” Should Mean Here

Use “adapter” for code that translates between the repo’s internal contracts and an external system’s wire format or lifecycle.

That means:

- Cloud provider adapter: internal `GenerateRequest` to provider API.
- Local backend adapter: internal generation request to a model server or inference engine.
- Provisioner adapter: internal `NodeSpec` to a vendor’s rental API.

## 5. Practical Recommendation

If you add more providers, keep the routing decision separate from the adapter implementation:

- router decides *which* provider tier to use,
- adapter decides *how* to talk to that provider,
- model manager decides *how* to serve the local model,
- provisioner decides *how* to rent or terminate hardware.

That separation is what keeps the control plane deterministic.
