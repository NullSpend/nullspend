export interface NullSpendErrorBody {
  error: { code: string; message: string; details: Record<string, unknown> | null };
}

export function errorResponse(
  code: string,
  message: string,
  status: number,
  details?: Record<string, unknown> | null,
): Response {
  const body: NullSpendErrorBody = {
    error: { code, message, details: details ?? null },
  };
  return Response.json(body, { status });
}

// ISSUE-034: SDK consumers and curl users default to Authorization: Bearer
// because it's the universal API convention. The proxy only accepts the
// x-nullspend-key header (matching the dashboard's lib/auth/api-key.ts
// ISSUE-008 fix). Surface a direct hint instead of the generic 401 so they
// can self-correct without grepping the docs.
export function unauthorizedResponse(request: Request): Response {
  if (!request.headers.get("x-nullspend-key") && request.headers.get("authorization")) {
    return errorResponse(
      "unauthorized",
      'Use the "x-nullspend-key" header to send your NullSpend API key. "Authorization: Bearer" is not accepted on this endpoint.',
      401,
    );
  }
  return errorResponse("unauthorized", "Invalid or missing authentication header", 401);
}
