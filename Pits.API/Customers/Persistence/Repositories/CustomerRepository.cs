using Mecanillama.API.Customers.Domain.Model;
using Mecanillama.API.Customers.Domain.Repositories;
using Mecanillama.API.Shared.Domain.Repositories;
using Mecanillama.API.Shared.Persistence.Contexts;
using Microsoft.EntityFrameworkCore;

namespace Mecanillama.API.Customers.Persistence.Repositories;

public class CustomerRepository :BaseRepository, ICustomerRepository
{
    public CustomerRepository(AppDbContext context) : base(context){}


    public async Task<IEnumerable<Customer>> ListAsync()
    {
        return await Context.Customers.ToListAsync();
    }

    public async Task AddAsync(Customer customer)
    {
        await Context.Customers.AddAsync(customer);
    }

    public async Task<Customer> FindByIdAsync(long id)
    {
        return await Context.Customers.FindAsync(id);
    }
    
    public async Task<Customer> FindByUserIdAsync(long userId)
    {
        return await Context.Customers
            .Where(p => p.UserId == userId)
            .FirstOrDefaultAsync();
    }

    public void Update(Customer customer)
    {
        Context.Customers.Update(customer);
    }

    public void Remove(Customer customer)
    {
        Context.Customers.Remove(customer);
    }
}