using AutoMapper;
using Mecanillama.API.Appointments.Domain.Models;
using Mecanillama.API.Appointments.Resources;
using Mecanillama.API.Customers.Domain.Model;
using Mecanillama.API.Customers.Resources;
using Mecanillama.API.Mechanics.Domain.Models;
using Mecanillama.API.Mechanics.Resources;
using Mecanillama.API.Reviews.Domain.Models;
using Mecanillama.API.Reviews.Resources;
using Mecanillama.API.Security.Domain.Models;
using Mecanillama.API.Security.Domain.Services.Communication;

namespace Mecanillama.API.Customers.Mapping;

public class ResourceToModelProfile : Profile
{
    public ResourceToModelProfile()
    {
        CreateMap<SaveCustomerResource, Customer>();
        CreateMap<SaveMechanicResource, Mechanic>();
        CreateMap<UpdateMechanicResource, Mechanic>();
        CreateMap<SaveAppointmentResource, Appointment>();
        CreateMap<SaveReviewResource, Review>();
        CreateMap<RegisterRequest, User>();

        CreateMap<UpdateRequest, User>()
            .ForAllMembers(options => options.Condition(
                (_, _, property) =>
                {
                    if (property == null) return false;
                    if (property is string && string.IsNullOrEmpty((string)property)) return false;
                    return true;
                }
            ));
    }
}