import type { NextConfig } from "next";

// the recorded public demo is a fully static export (no server at all); everything else is the Docker image's standalone server
const staticDemo = process.env.NEXT_PUBLIC_DEMO_MODE === "static";

const nextConfig: NextConfig = {
  output: staticDemo ? "export" : "standalone",
  // partial prerendering cannot be used in a static export
  cacheComponents: !staticDemo,
  partialPrefetching: !staticDemo,
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
