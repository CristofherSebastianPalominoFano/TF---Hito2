
using Mecanillama.API.Security.Authorization.Handlers.Interfaces;
using Mecanillama.API.Security.Domain.Services;

namespace Mecanillama.API.Security.Authorization.MIddleware;

public class JwtMiddleware
{
    private readonly RequestDelegate _next;


    public JwtMiddleware(RequestDelegate next)
    {
        _next = next;
    }

    public async Task Invoke(HttpContext context, IUserService userService, IJwtHandler handler)
    {
        var token = context.Request.Headers["Authorization"].FirstOrDefault()?.Split(" ").Last();

        if (token != null)
        {
            var userId = handler.ValidateToken(token);

            if (userId != null)
            {
                // On successful JWT validation, attach user info to context
                context.Items["User"] = await userService.GetByIdAsync(userId.Value);
            }
        }

        await _next(context);
    }
}