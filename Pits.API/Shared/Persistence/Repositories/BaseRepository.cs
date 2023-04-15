using Mecanillama.API.Shared.Persistence.Contexts;

namespace Mecanillama.API.Shared.Domain.Repositories;


public class BaseRepository
{
    protected readonly AppDbContext Context;

    protected BaseRepository (AppDbContext context)
    {
        Context = context;
    }
}
