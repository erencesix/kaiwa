// netlify/edge-functions/transcribe.js
// Receives a single audio chunk from the browser as multipart/form-data,
// forwards it to OpenAI Whisper server-side, returns verbose_json.
// The API key never touches the browser.

export default async (request) => {
  if (request.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method Not Allowed" }), { status: 405, headers: { "Content-Type": "application/json" } });
  }

  const OPENAI_KEY = Deno.env.get("OPENAI_API_KEY");
  if (!OPENAI_KEY) {
    return new Response(JSON.stringify({ error: "API key not configured." }), { status: 500, headers: { "Content-Type": "application/json" } });
  }

  let formData;
  try {
    formData = await request.formData();
  } catch (e) {
    return new Response(JSON.stringify({ error: "Invalid form data: " + e.message }), { status: 400, headers: { "Content-Type": "application/json" } });
  }

  const audioFile = formData.get("file");
  if (!audioFile) {
    return new Response(JSON.stringify({ error: "No audio file received." }), { status: 400, headers: { "Content-Type": "application/json" } });
  }

  // Forward to Whisper — pass the file straight through, no re-encoding
  const outForm = new FormData();
  outForm.append("file", audioFile);
  outForm.append("model", "whisper-1");
  outForm.append("response_format", "verbose_json");
  outForm.append("timestamp_granularities[]", "segment");

  let whisperRes;
  try {
    whisperRes = await fetch("https://api.openai.com/v1/audio/transcriptions", {
      method: "POST",
      headers: { Authorization: `Bearer ${OPENAI_KEY}` },
      body: outForm,
    });
  } catch (e) {
    return new Response(JSON.stringify({ error: "Failed to reach OpenAI: " + e.message }), { status: 503, headers: { "Content-Type": "application/json" } });
  }

  const rawText = await whisperRes.text();

  if (!whisperRes.ok) {
    let errMsg = "Whisper error (status " + whisperRes.status + ")";
    try { errMsg = JSON.parse(rawText).error?.message || errMsg; } catch {}
    return new Response(JSON.stringify({ error: errMsg }), { status: whisperRes.status, headers: { "Content-Type": "application/json" } });
  }

  // Pass the Whisper response straight back to the browser
  return new Response(rawText, {
    headers: { "Content-Type": "application/json" },
  });
};
