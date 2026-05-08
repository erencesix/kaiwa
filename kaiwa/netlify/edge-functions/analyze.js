// netlify/edge-functions/analyze.js
// Streams GPT-4o response via SSE — no timeout wall.

const JSON_HEADERS = { "Content-Type": "application/json" };

export default async (request) => {
  if (request.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method Not Allowed" }), { status: 405, headers: JSON_HEADERS });
  }

  const OPENAI_KEY = Deno.env.get("OPENAI_API_KEY");
  if (!OPENAI_KEY) {
    return new Response(JSON.stringify({ error: "API key not configured." }), { status: 500, headers: JSON_HEADERS });
  }

  let transcript, segments, language, detectedLanguage;
  try {
    const body = await request.json();
    transcript = body.transcript;
    segments = body.segments || [];
    language = body.language || "id";
    detectedLanguage = body.detectedLanguage || null;
  } catch (e) {
    return new Response(JSON.stringify({ error: "Invalid request: " + e.message }), { status: 400, headers: JSON_HEADERS });
  }

  if (!transcript || !transcript.trim()) {
    return new Response(JSON.stringify({ error: "Transcript is empty." }), { status: 400, headers: JSON_HEADERS });
  }

  const outputLangLabel =
    language === "en" ? "English" :
    language === "both" ? "Indonesian and English" :
    "Indonesian (Bahasa Indonesia)";

  const sourceLang = detectedLanguage || "the original language";
  const isSourceJapanese = detectedLanguage && detectedLanguage.toLowerCase().includes("japan");

  const segmentBlock = segments.length > 0
    ? segments.map(s => `[${s.startFormatted}] ${s.text}`).join("\n")
    : transcript;
  const cappedSegments = segmentBlock.length > 10000
    ? segmentBlock.substring(0, 10000) + "\n[... truncated ...]"
    : segmentBlock;

  const transcriptSchema = isSourceJapanese
    ? `"transcripts": {
    "translated": "[0:00] [Speaker A]: translated text",
    "ja": "[0:00] [Speaker A]: original japanese text",
    "both": "[0:00] [Speaker A - JP]: japanese\\n[0:00] [Speaker A - ID]: translated\\n\\n[0:05] [Speaker B - JP]: japanese\\n[0:05] [Speaker B - ID]: translated"
  }`
    : `"transcripts": {
    "translated": "[0:00] [Speaker A]: translated text\\n[0:05] [Speaker B]: translated text",
    "ja": "[0:00] [Speaker A]: original ${sourceLang} text\\n[0:05] [Speaker B]: original ${sourceLang} text",
    "both": "[0:00] [Speaker A - ORIG]: original ${sourceLang}\\n[0:00] [Speaker A - TRANS]: translated\\n\\n[0:05] [Speaker B - ORIG]: original ${sourceLang}\\n[0:05] [Speaker B - TRANS]: translated"
  }`;

  const systemPrompt = `You are a meeting analyst and translator. Analyze this meeting transcript and return a JSON object.

Source language: ${sourceLang}
Output language: ${outputLangLabel}

RULES — sections must be DISTINCT, no overlap:
- TRANSCRIPT: Every utterance timestamped. Format: "[M:SS] [Speaker A]: text". Include fillers and short responses.
- CHAPTERS: Time blocks by topic. Title = actual topic. No decisions here.
- KEY POINTS: Only explicit decisions, commitments, action items. WHO + WHAT. Nothing observational.
- HIGHLIGHTS: 2-4 verbatim quotes that are surprising or decisive. Explain why each matters.
- SUMMARY: Past-tense narrative. Specific names/numbers. Do NOT repeat key points or highlights verbatim.

Speaker detection: use real names if mentioned, else Speaker A/B/C. Be consistent.
Translation: natural and contextual, never literal. Match formality level.

Return ONLY this JSON, no markdown, no code fences:
{
  "speakers": [{"id":"speaker_a","label":"Speaker A","name":null,"role":null,"summary":"their specific contribution"}],
  "chapters": [{"title":"Actual Topic","timestamp":"0:00 - 2:30","summary":"what happened"}],
  "tabs": {
    "summary": [{"point":"theme with specific detail","subPoints":["detail","detail"]}],
    "keyPoints": [{"point":"WHO committed to WHAT","subPoints":["condition or deadline"]}],
    "highlights": [{"speaker":"Speaker A","quote":"exact translated quote","context":"why this matters"}]
  },
  ${transcriptSchema}
}`;

  let gptResponse;
  try {
    gptResponse = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${OPENAI_KEY}` },
      body: JSON.stringify({
        model: "gpt-4o",
        max_tokens: 4096,
        temperature: 0.1,
        stream: true,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: `Timestamped transcript:\n${cappedSegments}` },
        ],
      }),
    });
  } catch (err) {
    return new Response(JSON.stringify({ error: "Failed to reach OpenAI: " + err.message }), { status: 503, headers: JSON_HEADERS });
  }

  if (!gptResponse.ok) {
    const errText = await gptResponse.text();
    return new Response(JSON.stringify({ error: "OpenAI error: " + errText.substring(0, 300) }), { status: 502, headers: JSON_HEADERS });
  }

  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  const enc = new TextEncoder();

  const send = async (obj) => {
    try {
      await writer.write(enc.encode("data: " + JSON.stringify(obj) + "\n\n"));
    } catch (_) { /* writer closed on client disconnect */ }
  };

  (async () => {
    let accumulated = "";
    let lineBuf = ""; // proper SSE line buffer — survives TCP chunk boundaries

    try {
      const reader = gptResponse.body.getReader();
      const dec = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        lineBuf += dec.decode(value, { stream: true });

        let newlineIdx;
        while ((newlineIdx = lineBuf.indexOf("\n")) !== -1) {
          const line = lineBuf.slice(0, newlineIdx).trimEnd();
          lineBuf = lineBuf.slice(newlineIdx + 1);

          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (raw === "[DONE]") continue;

          let parsed;
          try { parsed = JSON.parse(raw); } catch { continue; }

          const token = parsed.choices?.[0]?.delta?.content;
          if (token) {
            accumulated += token;
            await send({ token });
          }
        }
      }

      const clean = accumulated
        .replace(/^```json\s*/, "").replace(/^```\s*/, "").replace(/\s*```$/, "").trim();

      let result;
      try {
        result = JSON.parse(clean);
      } catch (e) {
        await send({ error: "Failed to parse GPT response. Raw: " + clean.substring(0, 300) });
        return;
      }

      // Ensure all three transcript keys always exist
      if (result.transcripts) {
        result.transcripts.translated = result.transcripts.translated || "";
        result.transcripts.ja = result.transcripts.ja || result.transcripts.translated;
        result.transcripts.both = result.transcripts.both || result.transcripts.translated;
      }

      await send({ done: true, result });
    } catch (err) {
      await send({ error: "Stream error: " + err.message });
    } finally {
      try { await writer.close(); } catch (_) { /* already closed */ }
    }
  })();

  return new Response(readable, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "X-Accel-Buffering": "no",
    },
  });
};
