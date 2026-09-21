### Code Analysis & Verification Rules:
1. NEVER cite, reference, or deduce behavior from test files, mocks, or specs unless explicitly asked.
2. Treat all test files as potentially outdated, incomplete, or manufactured. 
3. Base all architectural decisions, logic verification, and bug explanations EXCLUSIVELY on the production/implementation code.
4. Walk through runtime execution paths step-by-step: trace inputs, state mutations, branch conditions, and return values directly from the source code.
5. Do not write or modify unit tests to "verify" your work. If you claim code works, justify it by tracing the actual implementation logic.