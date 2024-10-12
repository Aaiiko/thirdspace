
export type Business = {
    /** The name of the business */
    name : string;
    /** The description for the business */
    description : string;
    /** 1-5 Stars */
    review : number;
    /**  */
    price : number;
    /** Location */
    location : string;
    /** The catagory of restraunt (i.e. Greek, Pizza, etc.) */
    catagory : string;
    /** T */
    image : string|null;
    /** Business id */
    id : string;
}