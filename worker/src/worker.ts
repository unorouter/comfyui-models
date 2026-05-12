/**
 * R2 model proxy worker.
 *
 * Streams objects from a private R2 bucket to authenticated callers
 * (RunPod ComfyUI workers). No presigned URLs - the Worker holds the
 * R2 binding and uses a shared bearer token to gate reads.
 *
 * Routes:
 *   GET /get?key=<r2_key>   -> 200 bytes (streamed) | 404 | 401
 *   GET /head?key=<r2_key>  -> 200 with Content-Length header | 404 | 401
 *   GET /health             -> 200 ok
 *
 * Bindings (set in wrangler.toml):
 *   MODELS    -> R2 bucket binding (private)
 * Secrets:
 *   AUTH_TOKEN -> shared bearer token (matches R2_WORKER_TOKEN env in worker container)
 */

interface Env {
  MODELS: R2Bucket;
  AUTH_TOKEN: string;
}

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
  "Access-Control-Allow-Headers": "Authorization",
};

function unauthorized(): Response {
  return new Response("unauthorized", { status: 401, headers: CORS_HEADERS });
}

function badRequest(msg: string): Response {
  return new Response(msg, { status: 400, headers: CORS_HEADERS });
}

function notFound(): Response {
  return new Response("not found", { status: 404, headers: CORS_HEADERS });
}

function checkAuth(req: Request, env: Env): boolean {
  const header = req.headers.get("Authorization") || "";
  const expected = `Bearer ${env.AUTH_TOKEN}`;
  if (header.length !== expected.length) return false;
  let mismatch = 0;
  for (let i = 0; i < expected.length; i++) {
    mismatch |= header.charCodeAt(i) ^ expected.charCodeAt(i);
  }
  return mismatch === 0;
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    if (req.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    const url = new URL(req.url);
    if (url.pathname === "/health") {
      return new Response("ok", { status: 200, headers: CORS_HEADERS });
    }

    if (!checkAuth(req, env)) return unauthorized();

    const key = url.searchParams.get("key");
    if (!key) return badRequest("missing key");

    if (url.pathname === "/head") {
      const obj = await env.MODELS.head(key);
      if (!obj) return notFound();
      return new Response(null, {
        status: 200,
        headers: {
          ...CORS_HEADERS,
          "Content-Length": obj.size.toString(),
          "ETag": obj.httpEtag,
        },
      });
    }

    if (url.pathname === "/get") {
      const range = req.headers.get("Range");
      const opts: R2GetOptions = {};
      if (range) {
        const match = range.match(/^bytes=(\d+)-(\d*)$/);
        if (match) {
          const offset = Number(match[1]);
          const end = match[2] ? Number(match[2]) : undefined;
          opts.range = end !== undefined
            ? { offset, length: end - offset + 1 }
            : { offset };
        }
      }
      const obj = await env.MODELS.get(key, opts);
      if (!obj) return notFound();
      const headers: Record<string, string> = {
        ...CORS_HEADERS,
        "Content-Length": obj.size.toString(),
        "ETag": obj.httpEtag,
        "Accept-Ranges": "bytes",
      };
      if (range && obj.range) {
        const r = obj.range as { offset: number; length: number };
        const start = r.offset;
        const finish = r.offset + r.length - 1;
        headers["Content-Range"] = `bytes ${start}-${finish}/${obj.size}`;
        return new Response(obj.body, { status: 206, headers });
      }
      return new Response(obj.body, { status: 200, headers });
    }

    return notFound();
  },
};
