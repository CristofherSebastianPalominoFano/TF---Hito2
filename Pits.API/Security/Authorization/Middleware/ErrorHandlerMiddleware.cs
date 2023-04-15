using System.Net;
using System.Net.Mime;
using System.Text.Json;
using Mecanillama.API.Security.Exceptions;

namespace Mecanillama.API.Security.Authorization.MIddleware;

public class ErrorHandlerMiddleware
{
    private readonly RequestDelegate _next;

    public ErrorHandlerMiddleware(RequestDelegate next)
    {
        _next = next;
    }

    public async Task Invoke(HttpContext context)
    {
        try
        {
            await _next(context);
        }
        catch (Exception error)
        {
            var response = context.Response;
            response.ContentType = MediaTypeNames.Application.Json;

            response.StatusCode = error switch
            {
                AppException  =>
                    // Custom Application Exception
                    (int)HttpStatusCode.BadRequest,
                KeyNotFoundException =>
                    // Not found error
                    (int)HttpStatusCode.NotFound,
                _ => (int)HttpStatusCode.InternalServerError
            };

            var result = JsonSerializer.Serialize(new { message = error.Message });
            await response.WriteAsync(result);
        }
    }
}