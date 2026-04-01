import tseslint from "typescript-eslint";

const eslintConfig = [
  { ignores: [
    "packages/sdk/dist/**",
    "packages/mcp-server/dist/**",
    "packages/mcp-proxy/dist/**",
    "packages/cost-engine/dist/**",
    "packages/claude-agent/dist/**",
    "packages/db/dist/**",
    "packages/docs-mcp-server/dist/**",
    "apps/*/dist/**",
    "apps/*/.wrangler/**",
    "apps/*/worker-configuration.d.ts",
  ] },
  ...tseslint.configs.recommended,
  {
    rules: {
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_", caughtErrorsIgnorePattern: "^_" },
      ],
    },
  },
  {
    files: ["**/*.test.ts"],
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
    },
  },
];

export default eslintConfig;
