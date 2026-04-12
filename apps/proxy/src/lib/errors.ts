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
