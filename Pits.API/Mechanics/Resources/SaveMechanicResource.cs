using System.ComponentModel.DataAnnotations;

namespace Mecanillama.API.Mechanics.Resources;

public class SaveMechanicResource
{
    [Required]
    [MaxLength(200)]
    public string Name { get; set; }
    
    [Required]
    public long UserId { get; set; }

    public string Address { get; set; } = "undefined";

    public string Description { get; set; } = "undefined";
    
    public long Phone { get; set; }

}