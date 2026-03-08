// components/projects/CodeT5Project.jsx
// Drop this into your Next.js portfolio site

import { useState } from "react";

const TECH_STACK = [
  { label: "PyTorch", color: "bg-orange-100 text-orange-700 border-orange-200" },
  { label: "HuggingFace", color: "bg-yellow-100 text-yellow-700 border-yellow-200" },
  { label: "CodeT5", color: "bg-blue-100 text-blue-700 border-blue-200" },
  { label: "Transformers", color: "bg-purple-100 text-purple-700 border-purple-200" },
  { label: "Gradio", color: "bg-green-100 text-green-700 border-green-200" },
  { label: "CodeSearchNet", color: "bg-red-100 text-red-700 border-red-200" },
];

const METRICS = [
  { label: "BLEU-4", value: "~20", desc: "Code similarity score" },
  { label: "ROUGE-L", value: "0.38", desc: "Longest common subsequence" },
  { label: "Dataset", value: "412K", desc: "Python function-docstring pairs" },
  { label: "Model Params", value: "220M", desc: "CodeT5-base parameters" },
];

export default function CodeT5Project() {
  const [activeTab, setActiveTab] = useState("overview");

  return (
    <article className="rounded-2xl border border-gray-200 bg-white shadow-sm overflow-hidden hover:shadow-md transition-shadow duration-300">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-violet-600 p-6 text-white">
        <div className="flex items-start justify-between">
          <div>
            <span className="text-xs font-semibold uppercase tracking-widest text-indigo-200">
              Deep Learning · NLP · Code Generation
            </span>
            <h2 className="mt-1 text-xl font-bold leading-snug">
              CodeT5 Python Code Generator
            </h2>
            <p className="mt-1 text-sm text-indigo-100">
              Natural language → Python source code via fine-tuned Transformer
            </p>
          </div>
          <span className="rounded-full bg-white/20 px-3 py-1 text-xs font-medium">
            2024
          </span>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-100 text-sm font-medium">
        {["overview", "architecture", "results"].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-5 py-3 capitalize transition-colors ${
              activeTab === tab
                ? "border-b-2 border-indigo-600 text-indigo-600"
                : "text-gray-500 hover:text-gray-800"
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="p-6">
        {activeTab === "overview" && (
          <div className="space-y-5">
            <p className="text-sm text-gray-600 leading-relaxed">
              Fine-tuned{" "}
              <strong className="text-gray-900">Salesforce CodeT5</strong> on
              the CodeSearchNet Python corpus to generate syntactically correct
              Python functions from plain-English docstrings. The pipeline
              covers dataset loading, tokenization, Seq2Seq training with early
              stopping, and a Gradio inference UI.
            </p>

            {/* Input → Output demo */}
            <div className="rounded-xl bg-gray-50 border border-gray-200 overflow-hidden text-xs font-mono">
              <div className="flex items-center gap-2 bg-gray-100 px-4 py-2 text-gray-500 text-[11px]">
                <span className="h-2.5 w-2.5 rounded-full bg-red-400" />
                <span className="h-2.5 w-2.5 rounded-full bg-yellow-400" />
                <span className="h-2.5 w-2.5 rounded-full bg-green-400" />
                <span className="ml-2">demo</span>
              </div>
              <div className="p-4 space-y-3">
                <div>
                  <span className="text-indigo-500"># Input docstring</span>
                  <br />
                  <span className="text-gray-700">
                    &quot;Calculate the factorial of n recursively.&quot;
                  </span>
                </div>
                <div>
                  <span className="text-green-500"># Generated output</span>
                  <br />
                  <span className="text-gray-800">
                    {`def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)`
                      .split("\n")
                      .map((line, i) => (
                        <span key={i} className="block">
                          {line}
                        </span>
                      ))}
                  </span>
                </div>
              </div>
            </div>

            {/* Tech stack */}
            <div className="flex flex-wrap gap-2">
              {TECH_STACK.map(({ label, color }) => (
                <span
                  key={label}
                  className={`rounded-full border px-2.5 py-0.5 text-xs font-medium ${color}`}
                >
                  {label}
                </span>
              ))}
            </div>
          </div>
        )}

        {activeTab === "architecture" && (
          <div className="space-y-4 text-sm text-gray-600">
            <div className="rounded-xl border border-gray-200 bg-gray-50 p-4 font-mono text-xs text-gray-700 leading-relaxed">
              <pre>{`Docstring (text)
      │
      ▼
┌─────────────────────┐
│  RoBERTa Tokenizer  │  "Generate Python: ..."
└──────────┬──────────┘
           │ token ids
           ▼
┌─────────────────────┐
│   CodeT5 Encoder    │  12 layers · 768 hidden
└──────────┬──────────┘
           │ contextual embeddings
           ▼
┌─────────────────────┐
│   CodeT5 Decoder    │  12 layers · cross-attention
└──────────┬──────────┘
           │ token logits
           ▼
  Beam Search (k=5)
           │
           ▼
  Python source code`}</pre>
            </div>
            <ul className="space-y-2 list-disc list-inside text-gray-500 text-xs">
              <li>Task prefix conditioning: <code className="bg-gray-100 px-1 rounded">Generate Python:</code></li>
              <li>Padding tokens replaced with <code className="bg-gray-100 px-1 rounded">-100</code> so loss ignores them</li>
              <li>FP16 mixed-precision training on CUDA</li>
              <li>EarlyStoppingCallback (patience = 2 epochs)</li>
            </ul>
          </div>
        )}

        {activeTab === "results" && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              {METRICS.map(({ label, value, desc }) => (
                <div
                  key={label}
                  className="rounded-xl border border-gray-100 bg-gray-50 p-4"
                >
                  <p className="text-2xl font-bold text-indigo-600">{value}</p>
                  <p className="text-sm font-semibold text-gray-800">{label}</p>
                  <p className="text-xs text-gray-400 mt-0.5">{desc}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Footer links */}
      <div className="flex items-center gap-4 border-t border-gray-100 px-6 py-4 text-sm">
        <a
          href="https://github.com/YOUR_USERNAME/codet5-python-codegen"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 font-medium text-gray-700 hover:text-indigo-600 transition-colors"
        >
          {/* GitHub icon */}
          <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
          </svg>
          GitHub
        </a>
        <a
          href="#demo"
          className="flex items-center gap-1.5 font-medium text-gray-700 hover:text-indigo-600 transition-colors"
        >
          🚀 Live Demo
        </a>
        <a
          href="https://arxiv.org/abs/2109.00859"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 text-gray-400 hover:text-indigo-500 transition-colors text-xs ml-auto"
        >
          📄 CodeT5 Paper
        </a>
      </div>
    </article>
  );
}
