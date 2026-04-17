import { http, HttpResponse, type HttpHandler } from "msw";

export const ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages";

export function anthropicMessagesHandler(body: unknown): HttpHandler {
  return http.post(ANTHROPIC_MESSAGES_URL, () => HttpResponse.json(body));
}

export function anthropicErrorHandler(
  status: number,
  body: unknown,
): HttpHandler {
  return http.post(ANTHROPIC_MESSAGES_URL, () =>
    HttpResponse.json(body, { status }),
  );
}

export function anthropicNetworkErrorHandler(): HttpHandler {
  return http.post(ANTHROPIC_MESSAGES_URL, () => HttpResponse.error());
}

export function anthropicStreamingHandler(sseBody: string): HttpHandler {
  return http.post(
    ANTHROPIC_MESSAGES_URL,
    () =>
      new HttpResponse(sseBody, {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      }),
  );
}
