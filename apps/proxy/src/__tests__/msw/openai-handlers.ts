import { http, HttpResponse, type HttpHandler } from "msw";

export const OPENAI_CHAT_COMPLETIONS_URL =
  "https://api.openai.com/v1/chat/completions";

export function openaiChatCompletionHandler(body: unknown): HttpHandler {
  return http.post(OPENAI_CHAT_COMPLETIONS_URL, () => HttpResponse.json(body));
}

export function openaiErrorHandler(
  status: number,
  body: unknown,
): HttpHandler {
  return http.post(OPENAI_CHAT_COMPLETIONS_URL, () =>
    HttpResponse.json(body, { status }),
  );
}

export function openaiNetworkErrorHandler(): HttpHandler {
  return http.post(OPENAI_CHAT_COMPLETIONS_URL, () => HttpResponse.error());
}

export function openaiStreamingHandler(sseBody: string): HttpHandler {
  return http.post(
    OPENAI_CHAT_COMPLETIONS_URL,
    () =>
      new HttpResponse(sseBody, {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      }),
  );
}
