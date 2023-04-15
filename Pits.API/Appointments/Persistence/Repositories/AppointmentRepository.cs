using Mecanillama.API.Appointments.Domain.Models;
using Mecanillama.API.Appointments.Domain.Repositories;
using Mecanillama.API.Shared.Domain.Repositories;
using Mecanillama.API.Shared.Persistence.Contexts;
using Microsoft.EntityFrameworkCore;

namespace Mecanillama.API.Appointments.Persistence.Repositories;

public class AppointmentRepository: BaseRepository, IAppointmentRepository
{
    public AppointmentRepository(AppDbContext context) : base(context) { }

    public async Task<IEnumerable<Appointment>> ListAsync()
    {
        return await Context.Appointments.ToListAsync();
    }

    public async Task AddAsync(Appointment appointment)
    {
        if (Context.Appointments != null) await Context.Appointments.AddAsync(appointment);
    }

    public async Task<Appointment> FindByIdAsync(long id)
    {
        return await Context.Appointments.FindAsync(id);
    }

    public void Update(Appointment appointment)
    {
        Context.Appointments?.Update(appointment);
    }

    public void Remove(Appointment appointment)
    {
        Context.Appointments?.Remove(appointment);
    }
}