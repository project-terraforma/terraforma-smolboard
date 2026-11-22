const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");

const root = path.join(__dirname, "docs");
const port = Number(process.env.PORT || 4173);
const types = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
};

http.createServer((request, response) => {
  if (request.url === "/api/model-submissions" && request.method === "GET") {
    fs.readFile(path.join(root, "data", "model_submissions.json"), (error, data) => {
      if (error) {
        response.writeHead(500, { "Content-Type": "application/json; charset=utf-8" });
        response.end(JSON.stringify({ error: "Unable to read model submissions" }));
        return;
      }
      response.writeHead(200, { "Content-Type": "application/json; charset=utf-8" });
      response.end(data);
    });
    return;
  }

  if (request.url === "/api/model-submissions" && request.method === "POST") {
    let body = "";
    request.setEncoding("utf8");
    request.on("data", (chunk) => {
      body += chunk;
      if (body.length > 1_000_000) {
        request.destroy();
      }
    });
    request.on("end", () => {
      try {
        const submissions = JSON.parse(body);
        if (!Array.isArray(submissions)) {
          throw new Error("Submissions must be a JSON array");
        }
        fs.writeFile(
          path.join(root, "data", "model_submissions.json"),
          JSON.stringify(submissions, null, 2),
          (error) => {
            if (error) {
              response.writeHead(500, { "Content-Type": "application/json; charset=utf-8" });
              response.end(JSON.stringify({ error: "Unable to save model submissions" }));
              return;
            }
            response.writeHead(200, { "Content-Type": "application/json; charset=utf-8" });
            response.end(JSON.stringify({ ok: true }));
          },
        );
      } catch {
        response.writeHead(400, { "Content-Type": "application/json; charset=utf-8" });
        response.end(JSON.stringify({ error: "Invalid JSON" }));
      }
    });
    return;
  }

  const pathname = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
  const requested = pathname === "/" ? "/index.html" : pathname;
  const filePath = path.resolve(root, `.${requested}`);

  if (!filePath.startsWith(root + path.sep)) {
    response.writeHead(403).end("Forbidden");
    return;
  }

  fs.readFile(filePath, (error, data) => {
    if (error) {
      response.writeHead(404).end("Not found");
      return;
    }
    response.writeHead(200, { "Content-Type": types[path.extname(filePath)] || "application/octet-stream" });
    response.end(data);
  });
}).listen(port, "127.0.0.1", () => {
  console.log(`SMOLBoard is running at http://127.0.0.1:${port}`);
});
