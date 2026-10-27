import type { NextConfig } from "next";

// the recorded public demo is a fully static export (no server at all); everything else is the Docker image's standalone server
const staticDemo = process.env.NEXT_PUBLIC_DEMO_MODE === "static";
// the real app exported to static files, served by the FastAPI container itself (docker/fullstack.Dockerfile)
const exported = staticDemo || process.env.NEXT_PUBLIC_EXPORT === "1";

const nextConfig: NextConfig = {
  output: exported ? "export" : "standalone",
  // partial prerendering cannot be used in a static export
  cacheComponents: !exported,
  partialPrefetching: !exported,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
