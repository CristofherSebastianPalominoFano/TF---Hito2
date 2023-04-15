using System.ComponentModel.DataAnnotations;

namespace Mecanillama.API.Customers.Resources;

public class SaveCustomerResource
{
    [Required]
    [MaxLength(200)]
    public string Name { get; set; }
    
    [Required]
    public long UserId { get; set; }
}