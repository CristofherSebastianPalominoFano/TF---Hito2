
using Mecanillama.API.Security.Domain.Models;
using Mecanillama.API.Security.Domain.Repositories;
using Mecanillama.API.Shared.Domain.Repositories;
using Mecanillama.API.Shared.Persistence.Contexts;
using Microsoft.EntityFrameworkCore;

namespace Mecanillama.API.Security.Persistence.Repositories;

public class UserRepository: BaseRepository, IUserRepository
{
    public UserRepository(AppDbContext context) : base(context)
    {
    }

    public async Task<IEnumerable<User>> ListAsync()
    {
        return await Context.Users.ToListAsync();
    }

    public async Task AddAsync(User user)
    {
        await Context.Users.AddAsync(user);
    }

    public async Task<User> FindByIdAsync(long id)
    {
        return await Context.Users.FindAsync(id);
    }

    public async Task<User> FindByEmailAsync(string email)
    {
        return await Context.Users.SingleOrDefaultAsync(x => x.Email == email);
    }

    public bool ExistsByEmail(string email)
    {
        return Context.Users!.Any(x => x.Email == email);
    }

    public User? FindById(long id)
    {
        return Context.Users!.Find(id);
    }

    public void Update(User user)
    {
        Context.Users?.Update(user);
    }

    public void Remove(User user)
    {
        Context.Users?.Remove(user);
    }
    
}